"""思路 A 预处理脚本：构建效率地图 + 用问题一镜场验证查表精度。

用法：
    python3 build_map.py            # 建表 → 保存 npz → 验证 → 打印报告

产物：
    02_论文/结果/efficiency_map.npz   效率地图（查表数据源）
"""

import os
import time

import numpy as np
from efficiency_map import (
    EfficiencyMap,
    _attitude_cos_batch,
    build_efficiency_map,
    trunc_free_batch,
)
from q1_functions import (
    REP_HOURS,
    TRUNC_RAYS_DEFAULT,
    dni,
    load_mirrors,
    precompute_position_terms,
    sun_geometry,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATTACH_PATH = os.path.join(ROOT, "00_题目与数据", "附件.xlsx")
MAP_PATH = os.path.join(ROOT, "02_论文", "结果", "efficiency_map.npz")


def validate_against_q1(map_path, n_rays=None, grid_n=10):
    if n_rays is None:
        n_rays = TRUNC_RAYS_DEFAULT
    """用附件 1745 镜位置对比：直接追迹 vs 查表插值。

    基准（真值）：60 时刻逐镜直接光锥追迹的年平均效率；
    对比：效率地图双线性插值。
    输出 φ（能量加权）、η_cos、η_trunc 的平均/最大绝对误差。
    """
    mirror_xy, mirror_center = load_mirrors(ATTACH_PATH)
    em = EfficiencyMap(map_path)
    lookup = em.query(mirror_xy)

    # 直接追迹基准（与建表同参数，验证插值误差而非采样误差）
    phi_dir = np.zeros(len(mirror_center))
    cos_dir = np.zeros(len(mirror_center))
    trunc_dir = np.zeros(len(mirror_center))
    terms = precompute_position_terms(mirror_center)
    eta_at = terms["eta_at"]
    for month in range(1, 13):
        for hour in REP_HOURS:
            s = sun_geometry(month, hour)[4]
            d = dni(s[2])
            ec = _attitude_cos_batch(mirror_center, s)
            et = trunc_free_batch(mirror_center, s, n_rays, grid_n)
            phi_dir += d * ec * et * eta_at
            cos_dir += ec
            trunc_dir += et
    n_t = 12 * len(REP_HOURS)
    phi_dir, cos_dir, trunc_dir = phi_dir / n_t, cos_dir / n_t, trunc_dir / n_t

    report = {}
    for name, direct, table in (
            ("phi(能量加权)", phi_dir, lookup["phi"]),
            ("eta_cos_yr", cos_dir, lookup["eta_cos_yr"]),
            ("eta_trunc_yr", trunc_dir, lookup["eta_trunc_yr"])):
        err = np.abs(direct - table)
        report[name] = (float(err.mean()), float(err.max()),
                        float(np.median(err)))
    return report


def main():
    os.makedirs(os.path.dirname(MAP_PATH), exist_ok=True)

    t0 = time.time()
    print(f"开始构建效率地图（71×71 网格 × 60 时刻，每点 {TRUNC_RAYS_DEFAULT} 条光线）...")
    data = build_efficiency_map(save_path=MAP_PATH)
    print(f"建表完成: {time.time() - t0:.1f}s, 网格 "
          f"{len(data['ys'])}×{len(data['xs'])}, "
          f"有效位置 {int(data['mask'].sum())}/{data['mask'].size}")

    print("\n开始验证（附件 1745 镜直接追迹 vs 查表插值）...")
    t0 = time.time()
    report = validate_against_q1(MAP_PATH)
    print(f"验证完成: {time.time() - t0:.1f}s\n")
    print(f"{'字段':<14}{'平均绝对误差':>12}{'最大绝对误差':>12}{'中位数':>10}")
    for name, (mean_e, max_e, med_e) in report.items():
        print(f"{name:<16}{mean_e:>12.4f}{max_e:>12.4f}{med_e:>10.4f}")
    print(f"\n地图已保存: {MAP_PATH}")


if __name__ == "__main__":
    main()
