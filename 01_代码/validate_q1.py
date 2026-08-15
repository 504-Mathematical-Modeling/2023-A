"""问题一模型与数值验证（统一入口）。

验证项目：
  1. conv     数值收敛性（全部 60 代表时刻）：网格数 / 光线数 / 邻域半径
  2. neighbor 邻域半径：最不利时刻(12 月 9:00/15:00) vs 全镜场基准 + 60 时刻 20m vs 40m
  3. tower    塔身半径敏感性（2.0--4.0m，塔影采样点占比）
  4. trunc    截断效率三基准（镜心主光线 / 零锥角确定性 / 加密采样）
  5. symmetry 对称性检验（同月 9:00 vs 15:00 功率对比）
  6. dist     截断效率随镜塔水平距离分组（支撑论文 tab:trunc_distance）

用法：
    python validate_q1.py            # 全部运行
    python validate_q1.py conv       # 只运行指定项，可组合如 "conv symmetry"
    python validate_q1.py --list     # 列出可选项

输出（02_论文/files/）：
    验证_收敛60时刻.csv   —— conv
    验证_邻域半径.csv     —— neighbor
    验证_邻域60时刻.csv   —— neighbor
    验证_塔身半径.csv     —— tower
    验证_截断基准.csv     —— trunc
    验证_对称性.csv       —— symmetry
    截断效率随距离.csv    —— dist
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from q1_functions import (
    COLLECTOR_RADIUS,
    COLLECTOR_Z_HIGH,
    COLLECTOR_Z_LOW,
    MIRROR_AREA,
    REP_HOURS,
    SAMPLING_EPS,
    build_neighbors,
    collector_trunc_efficiency,
    dni,
    effective_points,
    field_power_mw,
    load_mirrors,
    mirror_attitude,
    mirror_optical_efficiency,
    precompute_position_terms,
    sun_geometry,
    tower_shadow_mask,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATTACH = os.path.join(ROOT, "00_题目与数据", "附件.xlsx")
OUTDIR = os.path.join(ROOT, "02_论文", "files")

# 邻域半径验证用时刻：冬季低太阳角最不利工况
WINTER_LOW_MOMENTS = [(12, 9.0), (12, 15.0)]
# 塔身半径敏感性用时刻：冬季低太阳角 + 夏季正午
TOWER_MOMENTS = [(12, 9.0), (12, 15.0), (6, 12.0)]
# 截断效率距离分组：镜塔水平距离区间 (m)
DIST_BINS = [(100, 150), (150, 200), (200, 250), (250, 300), (300, 350)]
# 截断距离分组用代表时刻
DIST_MOMENTS = [(6, 12.0), (12, 12.0), (3, 12.0), (6, 9.0)]


def load_field():
    """加载镜场数据并预计算位置相关量。"""
    _, mc = load_mirrors(ATTACH)
    area = np.full(len(mc), MIRROR_AREA)
    pt = precompute_position_terms(mc)
    return mc, area, pt


def one_moment(mc, area, pt, nb, month, hour, gn=10, nr=1024):
    """单个时刻的场平均光学效率、功率及分项效率。"""
    _, _, _, _, s = sun_geometry(month, hour)
    _, n, ec, _, ea = mirror_attitude(mc, s, pt)
    P, v = effective_points(mc, s, n, nb, gn)
    esb = v.mean(axis=1)
    et = collector_trunc_efficiency(P, s, n, v, nr)
    ei = mirror_optical_efficiency(ec, ea, eta_sb=esb, eta_trunc=et)
    eta = float(np.sum(area * ei) / np.sum(area))
    p = field_power_mw(dni(s[2]), area, ei)
    q = p * 1000.0 / float(np.sum(area))
    return eta, p, q, float(np.mean(esb)), float(np.mean(et))


def annual_mean(mc, area, pt, nb, gn, nr):
    """全部 60 个代表时刻的年平均光学效率与输出热功率。"""
    e_sum = p_sum = 0.0
    n = 0
    for m in range(1, 13):
        for h in REP_HOURS:
            e, p, _, _, _ = one_moment(mc, area, pt, nb, m, h, gn, nr)
            e_sum += e
            p_sum += p
            n += 1
    return e_sum / n, p_sum / n


def full_field_neighbors(mc):
    """无邻域限制（全镜场候选）的邻域索引。"""
    return [np.arange(len(mc), dtype=int) for _ in range(len(mc))]


# ============================ 1. 数值收敛性 ============================
SAVE_CONV = "验证_收敛60时刻.csv"


def run_convergence(mc, area, pt, series):
    """60 时刻收敛性：网格 / 光线 / 邻域半径三系列，带断点续跑。"""
    path = os.path.join(OUTDIR, SAVE_CONV)
    rows = []
    if os.path.exists(path):
        rows = pd.read_csv(path).to_dict("records")
    done = {(r["测试项"], r["参数值"]) for r in rows}

    def run(tag, val, gn, nr, rad):
        if (tag, str(val)) in done:
            print(f"[{tag}={val}] 已存在，跳过", flush=True)
            return
        t0 = time.time()
        nb = build_neighbors(mc, rad)
        e, p = annual_mean(mc, area, pt, nb, gn, nr)
        rows.append({"测试项": tag, "参数值": str(val),
                     "年均eta": round(e, 4), "年均功率(MW)": round(p, 3),
                     "耗时(s)": round(time.time() - t0, 1)})
        print(f"[{tag}={val}] eta={e*100:.4f}% P={p:.3f}MW "
              f"{time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")

    if series in ("grid", "all"):
        for gn in [8, 10, 15]:
            run("网格数", f"{gn}×{gn}", gn, 1024, 20.0)
    if series in ("rays", "all"):
        for nr in [512, 1024, 2048]:
            run("光线数", nr, 10, nr, 20.0)
    if series in ("radius", "all"):
        for rad in [20.0, 30.0, 40.0]:
            run("邻域半径", f"{rad:.0f}m", 10, 1024, rad)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"保存: {path}")
    print(df.to_string(index=False))


# ============================ 2. 邻域半径 ============================
def test_neighbor_radius(mc, area, pt):
    """最不利时刻（12 月 9:00/15:00）20/30/40/50m vs 全镜场基准。"""
    rows = []
    for rad in [20.0, 30.0, 40.0, 50.0]:
        nb = build_neighbors(mc, rad)
        avg = np.mean([len(x) for x in nb])
        for m, h in WINTER_LOW_MOMENTS:
            eta, p, _, esb, et = one_moment(mc, area, pt, nb, m, h)
            rows.append({"半径(m)": rad, "月": m, "时刻": h,
                         "平均候选数": round(avg, 1),
                         "eta_sb": round(esb, 4), "eta_trunc": round(et, 4),
                         "eta": round(eta, 4), "功率(MW)": round(p, 3)})
    nb = full_field_neighbors(mc)
    avg = len(mc) - 1
    for m, h in WINTER_LOW_MOMENTS:
        eta, p, _, esb, et = one_moment(mc, area, pt, nb, m, h)
        rows.append({"半径(m)": "全场", "月": m, "时刻": h,
                     "平均候选数": avg,
                     "eta_sb": round(esb, 4), "eta_trunc": round(et, 4),
                     "eta": round(eta, 4), "功率(MW)": round(p, 3)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "验证_邻域半径.csv"),
              index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"保存: {os.path.join(OUTDIR, '验证_邻域半径.csv')}")


def test_neighbor_60(mc, area, pt):
    """全部 60 时刻 20m vs 40m 年均对比。"""
    nb20 = build_neighbors(mc, 20.0)
    nb40 = build_neighbors(mc, 40.0)
    e20, p20 = annual_mean(mc, area, pt, nb20, 10, 1024)
    e40, p40 = annual_mean(mc, area, pt, nb40, 10, 1024)
    rows = [
        {"半径(m)": 20, "年均eta": round(e20, 4),
         "年均功率(MW)": round(p20, 3),
         "eta相对差(%)": "—", "功率相对差(%)": "—"},
        {"半径(m)": 40, "年均eta": round(e40, 4),
         "年均功率(MW)": round(p40, 3),
         "eta相对差(%)": round(abs(e40 - e20) / e20 * 100, 4),
         "功率相对差(%)": round(abs(p40 - p20) / p20 * 100, 4)},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "验证_邻域60时刻.csv"),
              index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"保存: {os.path.join(OUTDIR, '验证_邻域60时刻.csv')}")


# ============================ 3. 塔身半径敏感性 ============================
def test_tower_radius(mc, area, pt, nb):
    """塔身半径 2.0--4.0m 的塔影采样点占比（相对 3.5m 基准）。"""
    rows = []
    base = {}
    for m, h in TOWER_MOMENTS:
        _, _, _, _, s = sun_geometry(m, h)
        _, n, _, _, _ = mirror_attitude(mc, s, pt)
        P, _ = effective_points(mc, s, n, nb)
        S = P.shape[1]
        hit = tower_shadow_mask(P.reshape(-1, 3), s, radius=3.5,
                                z_lo=0.0, z_hi=84.0).reshape(-1, S)
        base[(m, h)] = float(hit.mean())
    for rt in [2.0, 2.5, 3.0, 3.5, 4.0]:
        for m, h in TOWER_MOMENTS:
            _, _, _, _, s = sun_geometry(m, h)
            _, n, _, _, _ = mirror_attitude(mc, s, pt)
            P, _ = effective_points(mc, s, n, nb)
            S = P.shape[1]
            hit = tower_shadow_mask(P.reshape(-1, 3), s, radius=rt,
                                    z_lo=0.0, z_hi=84.0).reshape(-1, S)
            frac = float(hit.mean())
            rows.append({"塔半径(m)": rt, "月": m, "时刻": h,
                         "塔影采样点占比": round(frac, 4),
                         "相对3.5m差(占比)": round(frac - base[(m, h)], 4)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "验证_塔身半径.csv"),
              index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"保存: {os.path.join(OUTDIR, '验证_塔身半径.csv')}")


# ============================ 4. 截断效率基准 ============================
def test_trunc_baseline(mc, pt, nb):
    """截断效率三基准：镜心主光线 / 零锥角确定性 / 加密采样。"""
    rows = []
    _, _, _, _, s = sun_geometry(6, 12.0)
    _, n, _, _, _ = mirror_attitude(mc, s, pt)

    # (1) 镜心主光线应精确指向集热器中心
    r_main = 2.0 * (s @ n.T)[:, None] * n - s[None, :]
    dot = np.sum(r_main * pt["t"], axis=1)
    err = np.max(np.abs(1.0 - dot))
    rows.append({"检验项": "镜心主光线命中集热器中心",
                 "说明": f"反射方向与目标方向夹角余弦最大偏差 {err:.2e}"})

    # (2) 零锥角确定性：镜心主光线必过集热器中心，命中率应为 100%
    centers = mc
    r_c = 2.0 * (s @ n.T)[:, None] * n - s[None, :]
    A = r_c[:, 0] ** 2 + r_c[:, 1] ** 2
    okA = A > 1e-12
    B = 2.0 * (centers[:, 0] * r_c[:, 0] + centers[:, 1] * r_c[:, 1])
    Cc = centers[:, 0] ** 2 + centers[:, 1] ** 2 - COLLECTOR_RADIUS ** 2
    disc = B ** 2 - 4.0 * A * Cc
    lam = (-B - np.sqrt(np.maximum(disc, 0))) / (2.0 * A)
    z_hit = centers[:, 2] + lam * r_c[:, 2]
    det_hit = okA & (disc >= 0) & (lam > SAMPLING_EPS) \
        & (z_hit >= COLLECTOR_Z_LOW) & (z_hit <= COLLECTOR_Z_HIGH)
    rows.append({"检验项": "零锥角确定性命中率",
                 "说明": f"{det_hit.mean()*100:.2f}%（应≈100%）"})

    # (3) 加密采样：前 30 面镜 grid10/1024 vs grid20/2048
    sel = np.arange(min(30, len(mc)))
    mc_sel = mc[sel]
    n_sel = n[sel]
    nb_sel = build_neighbors(mc_sel, 20.0)
    P10, v10 = effective_points(mc_sel, s, n_sel, nb_sel, 10)
    et10 = collector_trunc_efficiency(P10, s, n_sel, v10, 1024)
    P20, v20 = effective_points(mc_sel, s, n_sel, nb_sel, 20)
    et20 = collector_trunc_efficiency(P20, s, n_sel, v20, 2048)
    diff = np.abs(et10 - et20)
    rows.append({"检验项": "加密采样(30镜)",
                 "说明": f"grid10/1024 vs grid20/2048, "
                         f"最大差 {diff.max():.4f}, 平均差 {diff.mean():.4f}"})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "验证_截断基准.csv"),
              index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"保存: {os.path.join(OUTDIR, '验证_截断基准.csv')}")


# ============================ 5. 对称性检验 ============================
def test_symmetry(mc, area, pt, nb):
    """同月 9:00 与 15:00 输出热功率对比。"""
    rows = []
    for m in [3, 6, 9, 12]:
        _, p9, _, _, _ = one_moment(mc, area, pt, nb, m, 9.0)
        _, p15, _, _, _ = one_moment(mc, area, pt, nb, m, 15.0)
        rows.append({"月": m, "9:00功率(MW)": round(p9, 3),
                     "15:00功率(MW)": round(p15, 3),
                     "相对差(%)": round(abs(p9 - p15) / max(p9, p15) * 100, 2)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "验证_对称性.csv"),
              index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"保存: {os.path.join(OUTDIR, '验证_对称性.csv')}")


# ============================ 6. 截断效率随距离分组 ============================
def test_trunc_by_distance(mc, pt, nb):
    """截断效率随镜塔水平距离分组统计（支撑论文 tab:trunc_distance）。"""
    d_h = np.hypot(mc[:, 0], mc[:, 1])
    rows = []
    for m, h in DIST_MOMENTS:
        _, _, _, _, s = sun_geometry(m, h)
        _, n, _, _, _ = mirror_attitude(mc, s, pt)
        P, valid = effective_points(mc, s, n, nb)
        et = collector_trunc_efficiency(P, s, n, valid)
        for lo, hi in DIST_BINS:
            sel = (d_h >= lo) & (d_h < hi)
            if sel.sum() == 0:
                continue
            rows.append({"时刻": f"{m}月{h:g}时", "距离组": f"{lo}-{hi} m",
                         "镜面数": int(sel.sum()),
                         "平均截断效率": round(float(et[sel].mean()), 4)})
        print(f"  {m}月{h:g}时 完成", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "截断效率随距离.csv"),
              index=False, encoding="utf-8-sig")
    print(df.pivot(index="距离组", columns="时刻",
                   values="平均截断效率").to_string())
    print(f"保存: {os.path.join(OUTDIR, '截断效率随距离.csv')}")


# ============================ 主入口 ============================
PARTS = {
    "conv": run_convergence,
    "neighbor": None,   # 组合函数，见 main 内分发
    "tower": test_tower_radius,
    "trunc": test_trunc_baseline,
    "symmetry": test_symmetry,
    "dist": test_trunc_by_distance,
}


def main():
    parser = argparse.ArgumentParser(
        description="问题一模型与数值验证（统一入口）")
    parser.add_argument("parts", nargs="*", default=["all"],
                        help="验证项：conv neighbor tower trunc symmetry dist；"
                             "默认 all")
    parser.add_argument("--list", action="store_true", help="列出可选项")
    args = parser.parse_args()

    if args.list:
        print("可选项：conv neighbor tower trunc symmetry dist (all)")
        return

    os.makedirs(OUTDIR, exist_ok=True)
    mc, area, pt = load_field()
    nb = build_neighbors(mc, 20.0)

    which = args.parts if args.parts != ["all"] else list(PARTS)
    for name in which:
        t0 = time.time()
        print(f"\n===== {name} =====", flush=True)
        if name == "conv":
            run_convergence(mc, area, pt, "all")
        elif name == "neighbor":
            test_neighbor_radius(mc, area, pt)
            test_neighbor_60(mc, area, pt)
        elif name == "tower":
            test_tower_radius(mc, area, pt, nb)
        elif name == "trunc":
            test_trunc_baseline(mc, pt, nb)
        elif name == "symmetry":
            test_symmetry(mc, area, pt, nb)
        elif name == "dist":
            test_trunc_by_distance(mc, pt, nb)
        else:
            print(f"未知项: {name}（可选: conv neighbor tower trunc symmetry dist）")
        print(f"耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()