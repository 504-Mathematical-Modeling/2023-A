"""问题二结果验证：约束校验 + 月均功率表 + 60 时刻明细 + 相位稳健性。

用法：在 DE 主流程完成后运行（需 q2_result.npy 与 q2_sel.npy）。
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q1_functions import sun_geometry, dni, REP_HOURS
from common import (OUTPUT_DIR, FIGURE_DIR, hexagonal_lattice,
                    greedy_select, field_eval, param_neighbors,
                    param_terms, param_attitude, param_effective_points,
                    param_trunc, ETA_REF,
                    W_MIN, W_MAX, A_MIN, A_MAX, FIELD_RADIUS,
                    EXCLUSION_RADIUS, LATTICE_MARGIN)

REVIEW_GRID, REVIEW_RAYS = 8, 512

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC",
                                   "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_result():
    summ = np.load(os.path.join(OUTPUT_DIR, "q2_result.npy"),
                   allow_pickle=True).item()
    sel = np.load(os.path.join(OUTPUT_DIR, "q2_sel.npy"))
    return summ, sel


def check_constraints(summ, sel):
    """逐条核对题目约束，返回 (通过?, 明细列表)。"""
    tower, w, H, a = summ["tower"], summ["w"], summ["H"], summ["a"]
    d_min = w + LATTICE_MARGIN
    checks = []
    n = len(sel)

    r0 = np.hypot(sel[:, 0], sel[:, 1])
    bad = r0 > FIELD_RADIUS + 1e-6
    checks.append(("场地圆 |p| ≤ 350",
                   f"max|p| = {r0.max():.3f} m, 越界 {bad.sum()} 面",
                   bad.sum() == 0))

    rt = np.hypot(sel[:, 0] - tower[0], sel[:, 1] - tower[1])
    bad = rt < EXCLUSION_RADIUS - 1e-6
    checks.append(("禁装区 |p−T| ≥ 100",
                   f"min|p−T| = {rt.min():.3f} m, 违规 {bad.sum()} 面",
                   bad.sum() == 0))

    neigh = param_neighbors(sel, d_min + 1e-6)
    min_d = float("inf")
    viol = 0
    for i in range(n):
        if len(neigh[i]) == 0:
            continue
        dd = np.hypot(sel[neigh[i], 0] - sel[i, 0],
                      sel[neigh[i], 1] - sel[i, 1])
        m = dd.min()
        min_d = min(min_d, m)
        viol += int((dd < d_min - 1e-6).sum())
    checks.append(("相邻间距 ≥ w+5", f"min 间距 = {min_d:.3f} m "
                   f"(要求 {d_min:.2f}), 违规 {viol} 对", viol == 0))

    ok_dim = (W_MIN <= H <= w <= W_MAX) and (A_MIN <= a <= A_MAX) \
        and (a >= H / 2 - 1e-6)
    checks.append(("尺寸约束 2≤H≤w≤8, 2≤a≤6, a≥H/2",
                   f"w={w:.2f}, H={H:.2f}, a={a:.2f}",
                   ok_dim))

    ok_tower = np.hypot(*tower) <= FIELD_RADIUS
    checks.append(("塔位在场内 |T| ≤ 350",
                   f"|T| = {np.hypot(*tower):.2f} m", ok_tower))
    return checks


def monthly_table(summ, sel):
    """12 个月 × 5 时刻 等权平均口径：逐时刻功率、逐月平均、年均。"""
    tower, w, H, a = summ["tower"], summ["w"], summ["H"], summ["a"]
    mc = np.column_stack([sel, np.full(len(sel), a)])
    area = w * H
    neigh = param_neighbors(sel, w + LATTICE_MARGIN + 5.0)
    terms = param_terms(mc, tower)
    grid_n, n_rays = REVIEW_GRID, REVIEW_RAYS
    rows = []
    for month in range(1, 13):
        for hour in REP_HOURS:
            s = sun_geometry(month, hour)[4]
            _, n, eta_cos, _, eta_at = param_attitude(mc, s, tower, terms)
            P, valid = param_effective_points(mc, s, n, neigh, w, H,
                                              tower, grid_n)
            eta_sb = valid.mean(axis=1)
            eta_trunc = param_trunc(P, s, n, valid, tower, n_rays)
            eta_i = ETA_REF * eta_cos * eta_at * eta_sb * eta_trunc
            p = dni(s[2]) * area * eta_i.sum() / 1000.0
            rows.append({"月份": month, "时刻": f"{hour:.1f}h",
                         "DNI": dni(s[2]), "P_MW": p,
                         "eta_cos": eta_cos.mean(), "eta_sb": eta_sb.mean(),
                         "eta_trunc": eta_trunc.mean(),
                         "eta": eta_i.mean()})
    df = pd.DataFrame(rows)
    mon = df.groupby("月份")["P_MW"].mean().round(3)
    p_avg = df["P_MW"].mean()
    return df, mon, p_avg


def phase_robustness(summ, sel):
    """用不同晶格相位重建贪心布局，检查功率口径稳健性（±0.5% 内视为稳）。"""
    tx, ty, w, H, a = (summ["tower"][0], summ["tower"][1],
                       summ["w"], summ["H"], summ["a"])
    x = summ["x"]
    theta, phx, phy = x[5], x[6], x[7]
    d = w + LATTICE_MARGIN
    base = summ["P_mw"]
    out = []
    for ph in [(0., 0.), (2., 2.), (4., 4.), (6., 6.), (8., 8.),
               (1.3, 5.7), (9., 3.)]:
        cand = hexagonal_lattice(np.array([tx, ty]), d, theta, ph)
        s2, _, _ = greedy_select(cand, np.array([tx, ty]), w, H, a)
        P2, _ = field_eval(s2, np.array([tx, ty]), w, H, a,
                           REVIEW_GRID, REVIEW_RAYS)
        out.append((ph, len(s2), P2, (P2 - base) / base * 100))
    return out


def plot_monthly(df, save_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    piv = df.pivot(index="月份", columns="时刻", values="P_MW")
    for col in piv.columns:
        ax.plot(piv.index, piv[col], marker="o", ms=4,
                label=f"{col} 时")
    ax.axhline(df["P_MW"].mean(), color="k", ls="--", lw=1,
               label=f"年均 {df['P_MW'].mean():.2f} MW")
    ax.set_xlabel("月份"); ax.set_ylabel("代表时刻输出热功率 (MW)")
    ax.set_xticks(range(1, 13))
    ax.legend(ncol=3, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def main():
    summ, sel = load_result()
    print("=" * 64)
    print("问题二结果验证")
    print(f"塔位 ({summ['tower'][0]:.1f}, {summ['tower'][1]:.1f}) m, "
          f"镜 {summ['w']:.2f}×{summ['H']:.2f} m², a={summ['a']:.2f} m, "
          f"N={summ['N']}, A={summ['A_total']:.1f} m², "
          f"P̄={summ['P_mw']:.3f} MW, q={summ['q']:.4f} kW/m²")
    print("-" * 64)
    print("【1】约束校验")
    all_ok = True
    for name, detail, ok in check_constraints(summ, sel):
        all_ok &= ok
        print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    print(f"  → 全部通过: {all_ok}")

    print("-" * 64)
    print("【2】月均功率表（每月 5 时刻均值）")
    df, mon, p_avg = monthly_table(summ, sel)
    for m, v in mon.items():
        print(f"  {m:>2} 月: {v:6.3f} MW")
    print(f"  年均: {p_avg:.3f} MW (与 q2_result 的 {summ['P_mw']:.3f} 对比)")

    fig = plot_monthly(df, os.path.join(FIGURE_DIR, "fig_q2_monthly.png"))
    print(f"  月均图 → {fig}")

    print("-" * 64)
    print("【3】相位稳健性（不同晶格相位重建，P̄ 偏差 %）")
    for ph, n2, P2, dev in phase_robustness(summ, sel):
        print(f"  ph={ph}: N={n2}, P̄={P2:.2f} MW, 偏差 {dev:+.2f}%")

    print("-" * 64)
    df.to_csv(os.path.join(OUTPUT_DIR, "q2_timestep_table.csv"),
              index=False, float_format="%.4f")
    print(f"  60 时刻明细 → {os.path.join(OUTPUT_DIR, 'q2_timestep_table.csv')}")
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)