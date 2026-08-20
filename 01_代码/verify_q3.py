#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题三结果验证：约束校验 + 月均功率表"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC",
                                   "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q3_optimizer as q3
from common import OUTPUT_DIR, FIGURE_DIR

REVIEW_GRID, REVIEW_RAYS = 8, 512


def load_result():
    summ = np.load(os.path.join(OUTPUT_DIR, "q3_result.npy"), allow_pickle=True).item()
    sel = np.load(os.path.join(OUTPUT_DIR, "q3_sel.npy"))
    return summ, sel


def check_constraints(summ, sel):
    tower = np.asarray(summ["tower"], float)
    w = np.asarray(summ["w"], float)
    H = np.asarray(summ["H"], float)
    a = np.asarray(summ["a"], float)
    ok = True

    r = np.hypot(sel[:, 0], sel[:, 1])
    print(f"[1] 场地圆: max|p| = {r.max():.3f} m (≤350)  "
          f"{'✓' if r.max() <= 350 else '✗'}")

    rt = np.hypot(sel[:, 0] - tower[0], sel[:, 1] - tower[1])
    print(f"[2] 禁装区: min|p−T| = {rt.min():.3f} m (≥100)  "
          f"{'✓' if rt.min() >= 100 else '✗'}")

    lim = np.maximum(w[:, None], w[None, :]) + 5.0
    tree = cKDTree(sel)
    pairs = tree.query_pairs(lim.max())
    bad = []
    for i, j in pairs:
        if np.hypot(sel[i, 0] - sel[j, 0], sel[i, 1] - sel[j, 1]) < lim[i, j] - 1e-6:
            bad.append((i, j))
    print(f"[3] 相邻间距: 违规 {len(bad)} 对 (要求 ≥max(w_i,w_j)+5)  "
          f"{'✓' if not bad else '✗ 首次: ' + str(bad[0])}")
    if bad:
        ok = False

    bad_sz = ((H < 2) | (H > 8) | (w < H) | (w > 8) | (a < 2) | (a > 6) | (a < H / 2)).sum()
    print(f"[4] 尺寸约束: 违规 {bad_sz} 面 (2≤H≤w≤8, 2≤a≤6, a≥H/2)  "
          f"{'✓' if bad_sz == 0 else '✗'}")
    if bad_sz:
        ok = False

    t_in = np.hypot(tower[0], tower[1]) <= 350 - 100
    print(f"[5] 塔位在场: |T| = {np.hypot(tower[0], tower[1]):.2f} m "
          f"(禁装区外且在场内)  {'✓' if t_in else '✗'}")
    return ok


def monthly_table(summ, sel, grid_n=REVIEW_GRID, n_rays=REVIEW_RAYS):
    tower = np.asarray(summ["tower"], float)
    w = np.asarray(summ["w"], float)
    H = np.asarray(summ["H"], float)
    a = np.asarray(summ["a"], float)
    rows = []
    for m in range(1, 13):
        P, eff = q3.month_eval(sel, tower, w, H, a, m, grid_n, n_rays)
        A = float(np.sum(w * H))
        rows.append((m, eff["eta_cos"], eff["eta_sb"], eff["eta_trunc"],
                     eff["eta"], P, P * 1000.0 / A))
        print(f"  月 {m:2d}: P={P:7.3f} MW  η_cos={eff['eta_cos']:.4f} "
              f"η_sb={eff['eta_sb']:.4f} η_trunc={eff['eta_trunc']:.4f} "
              f"η={eff['eta']:.4f} q={P*1000.0/A:.4f}")
    df = pd.DataFrame(rows, columns=["月份", "eta_cos", "eta_sb", "eta_trunc",
                                     "eta", "p_mw", "q"])
    df.to_csv(os.path.join(OUTPUT_DIR, "q3_timestep_table.csv"), index=False)
    print(f"  年均: P̄ = {df['p_mw'].mean():.3f} MW (参考 q3_result: "
          f"{summ['P_mw']:.3f})")
    return df


def plot_monthly(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["月份"], df["p_mw"], "o-", color="#1f77b4", label="月均输出功率")
    ax.axhline(df["p_mw"].mean(), color="r", ls="--", lw=1,
               label=f"年均 {df['p_mw'].mean():.2f} MW")
    ax.set_xlabel("月份")
    ax.set_ylabel("输出热功率 (MW)")
    ax.set_xticks(range(1, 13))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "fig_q3_monthly.png"), dpi=150)
    plt.close(fig)
    print(f"  fig_q3_monthly.png → {FIGURE_DIR}")


def main():
    summ, sel = load_result()
    print("======== 问题三结果验证 ========")
    print(f"  塔位 ({summ['tower'][0]:.1f}, {summ['tower'][1]:.1f}), "
          f"N={summ['N']}, A={summ['A_total']:.1f} m², "
          f"P̄={summ['P_mw']:.3f} MW, q={summ['q']:.4f}")
    ok = check_constraints(summ, sel)
    print("\n[月均功率表] (8×8/512 口径)")
    df = monthly_table(summ, sel)
    plot_monthly(df)
    print("\n结论:", "全部通过 ✓" if ok and abs(df["p_mw"].mean() - summ["P_mw"]) / summ["P_mw"] < 0.01
          else "存在问题 ✗")


if __name__ == "__main__":
    main()