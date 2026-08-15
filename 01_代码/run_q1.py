"""问题一主流程：读附件 → 60 时刻循环 → 汇总 → 导出表格/图。

当前进度：
- 数据导入、质量检查、太阳位置、DNI、镜面姿态、解析效率。
- 阴影遮挡效率（塔阴影 + 镜间阴影/挡光，10×10 采样网格）。
- 截断效率（Sobol 准随机光锥光线追迹，1024 光线/点，圆柱 [76,84]m）。
- 导出：60 时刻明细 CSV、表 1/表 2 Excel、布局图与月平均变化图。
"""

import os

import numpy as np
import pandas as pd
from q1_functions import (
    MIRROR_AREA,
    REP_HOURS,
    build_neighbors,
    check_data_quality,
    collector_trunc_efficiency,
    dni,
    effective_points,
    eta_field_weighted,
    field_power_mw,
    load_mirrors,
    mirror_attitude,
    mirror_optical_efficiency,
    plot_mirror_layout,
    plot_monthly_metrics,
    precompute_position_terms,
    sun_geometry,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATTACH_PATH = os.path.join(ROOT, "00_题目与数据", "附件.xlsx")
FIGURE_DIR = os.path.join(ROOT, "02_论文", "figures")
OUTPUT_DIR = os.path.join(ROOT, "02_论文", "files")

DETAIL_COLS = ["月份", "时刻", "DNI", "eta_cos", "eta_sb", "eta_trunc",
               "eta", "p_mw", "q"]


def compute_hour_row(month, hour, mirror_center, area, position_terms, neighbors):
    """计算单个时刻的场平均指标，返回 dict（供 60 时刻明细使用）。"""
    _, _, _, _, s = sun_geometry(month, hour)
    _, n, eta_cos, _, eta_at = mirror_attitude(mirror_center, s, position_terms)

    P, valid = effective_points(mirror_center, s, n, neighbors)
    eta_sb = valid.mean(axis=1)
    eta_trunc = collector_trunc_efficiency(P, s, n, valid)
    eta_i = mirror_optical_efficiency(eta_cos, eta_at, eta_sb=eta_sb,
                                      eta_trunc=eta_trunc)
    dni_val = dni(s[2])
    p_mw = field_power_mw(dni_val, area, eta_i)

    return {
        "月份": month,
        "时刻": hour,
        "DNI": dni_val,
        "eta_cos": float(np.mean(eta_cos)),
        "eta_sb": float(np.mean(eta_sb)),
        "eta_trunc": float(np.mean(eta_trunc)),
        "eta": eta_field_weighted(eta_i, area),
        "p_mw": p_mw,
        "q": p_mw * 1000.0 / float(np.sum(area)),  # MW→kW / 总面积
    }


def build_tables(detail_df):
    """由 60 时刻明细生成表 1（每月）与表 2（年平均）。

    表 1：每月 21 日 5 时刻等权平均。
    表 2：12 个月等权平均。
    """
    mean_cols = ["eta_cos", "eta_sb", "eta_trunc", "eta", "p_mw", "q"]
    df1 = (detail_df.groupby("月份", sort=True)[mean_cols].mean().reset_index())
    df1["日期"] = df1["月份"].map(lambda m: f"{m} 月 21 日")
    df1 = df1[["日期"] + mean_cols]

    annual = {k: float(df1[k].mean()) for k in mean_cols}
    return df1, annual


def main():
    # 1. 数据加载与质量检查
    mirror_xy, mirror_center = load_mirrors(ATTACH_PATH)
    area = np.full(len(mirror_center), MIRROR_AREA)
    print(f"定日镜总数: {len(mirror_center)}")

    qc = check_data_quality(mirror_xy)
    print("数据质量检查:")
    print(f"  缺失/非数值: {'有' if qc['has_nan'] else '无'}"
          f", 重复坐标: {qc['duplicates']}")
    print(f"  水平距离范围: [{qc['r_min']:.1f}, {qc['r_max']:.1f}] m"
          f" (应 100~350), 超范围: {qc['out_of_range']}")
    print(f"  最近邻最小间距: {qc['min_spacing']:.2f} m (应 > 11)")

    # 2. 位置相关量缓存与候选镜筛选
    position_terms = precompute_position_terms(mirror_center)
    neighbors = build_neighbors(mirror_center)
    print(f"候选遮挡镜统计: 平均 {np.mean([len(nb) for nb in neighbors]):.1f} 面/镜")

    # 3. 60 时刻循环
    detail_rows = []
    for month in range(1, 13):
        for hour in REP_HOURS:
            detail_rows.append(compute_hour_row(
                month, hour, mirror_center, area, position_terms, neighbors))
    detail_df = pd.DataFrame(detail_rows, columns=DETAIL_COLS)

    # 4. 表 1、表 2
    df1, annual = build_tables(detail_df)

    print("\n======== 表 1：每月 21 日平均光学效率及输出功率 ========")
    print(df1.round(4).to_string(index=False))

    print("\n======== 表 2：年平均 ========")
    print({k: round(v, 4) for k, v in annual.items()})

    # 5. 导出中间结果与表格（02_论文/结果/）
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    detail_df.to_csv(os.path.join(OUTPUT_DIR, "60时刻明细.csv"),
                     index=False, encoding="utf-8-sig")
    df1.to_csv(os.path.join(OUTPUT_DIR, "表1_每月平均.csv"),
               index=False, encoding="utf-8-sig")
    annual_df = pd.DataFrame([annual])
    annual_df.to_csv(os.path.join(OUTPUT_DIR, "表2_年平均.csv"),
                     index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(os.path.join(OUTPUT_DIR, "问题一结果.xlsx")) as writer:
        df1.to_excel(writer, sheet_name="表1_每月平均", index=False)
        annual_df.to_excel(writer, sheet_name="表2_年平均", index=False)
        detail_df.to_excel(writer, sheet_name="60时刻明细", index=False)
    print(f"\n结果已导出到: {OUTPUT_DIR}")

    # 6. 绘图（02_论文/figures/）
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plot_mirror_layout(
        mirror_xy, save_path=os.path.join(FIGURE_DIR, "fig0_mirror_layout.png")
    )
    plot_monthly_metrics(
        df1, save_path=os.path.join(FIGURE_DIR, "fig1_monthly_metrics.png")
    )
    print(f"图片已保存到: {FIGURE_DIR}")

    return detail_df, df1, annual


if __name__ == "__main__":
    main()
