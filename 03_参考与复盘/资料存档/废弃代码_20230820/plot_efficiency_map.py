"""生成"快速效率地图"示意图（论文 5.2.4 小节）。

思路：左图展示全场预计算的年平均能量加权效率 Φ 地图（热力图 +
场地圆/塔周禁装区 + 吸收塔 + 六角晶格候选镜位）；
右图局部放大塔位附近区域，示意"网格预计算 → 双线性插值查表"：
候选镜位落在网格单元内，其贡献由四角格点值插值得到，
从而优化迭代中免去逐镜 60 时刻光锥追迹。

数据来源：02_论文/结果/efficiency_map.npz（build_map.py 产物）。
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from q1_functions import _set_cjk_font
from q2_optimizer import hexagonal_lattice

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_PATH = os.path.join(ROOT, "02_论文", "结果", "efficiency_map.npz")
FIG_PATH = os.path.join(ROOT, "02_论文", "figures", "fig2_emap.png")

FIELD_RADIUS = 350.0
EXCLUSION_RADIUS = 100.0
TOWER_XY = np.array([52.4, -118.6])      # 问题二优化塔位
GRID_STEP = 10.0                          # 效率地图网格步长（m）


def main():
    data = np.load(MAP_PATH)
    xs, ys = data["xs"], data["ys"]
    phi = data["phi"]                     # (ny, nx) 能量加权年平均效率

    _set_cjk_font()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # ---------- 左图：全场效率地图 ----------
    ax = axes[0]
    X, Y = np.meshgrid(xs, ys)
    im = ax.pcolormesh(X, Y, phi, cmap="YlOrRd", shading="auto", vmin=0.20,
                       vmax=0.62)
    ax.add_patch(Circle((0, 0), FIELD_RADIUS, fill=False, edgecolor="k",
                        linestyle="--", linewidth=1.0))
    ax.add_patch(Circle((0, 0), EXCLUSION_RADIUS, fill=False, edgecolor="#555",
                        linestyle=":", linewidth=1.2))
    ax.plot(*TOWER_XY, marker="*", ms=14, color="tab:blue", zorder=6,
            markeredgecolor="k", markeredgewidth=0.5)
    # 六角晶格候选镜位（问题二优化参数：d = w+5 = 10.8 m）
    cand = hexagonal_lattice(TOWER_XY, d=10.8, theta=0.0, phase=(0.0, 0.0))
    ax.plot(cand[::3, 0], cand[::3, 1], ".", ms=1.6, color="k", alpha=0.35,
            zorder=5)
    ax.set_xlim(-370, 370)
    ax.set_ylim(-370, 370)
    ax.set_aspect("equal")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("(a) 全场年平均能量加权效率地图 $\\Phi(p)$")
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("$\\Phi$ / (kW/m$^2$)")
    ax.text(0.99, 0.02, "实线：场地边界 350 m\n虚线：塔周禁装区 100 m\n"
            "★：吸收塔\n黑点：六角晶格候选镜位",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999", lw=0.6))

    # ---------- 右图：局部放大，示意网格预计算与插值查表 ----------
    ax = axes[1]
    x0, x1 = TOWER_XY[0] - 60, TOWER_XY[0] + 60
    y0, y1 = TOWER_XY[1] - 60, TOWER_XY[1] + 60
    im2 = ax.pcolormesh(X, Y, phi, cmap="YlOrRd", shading="auto", vmin=0.20,
                        vmax=0.62)
    ax.add_patch(Circle((0, 0), EXCLUSION_RADIUS, fill=False, edgecolor="#555",
                        linestyle=":", linewidth=1.2))
    ax.plot(*TOWER_XY, marker="*", ms=16, color="tab:blue", zorder=6,
            markeredgecolor="k", markeredgewidth=0.5)

    # 网格单元示意：画一个 2×2 单元的网格框
    base = np.array([np.ceil((x0 + 1) / GRID_STEP) * GRID_STEP,
                     np.ceil((y0 + 1) / GRID_STEP) * GRID_STEP])
    grid_x = base[0] + np.arange(-1, 3) * GRID_STEP
    grid_y = base[1] + np.arange(-1, 3) * GRID_STEP
    ax.plot(np.repeat(grid_x, 2), np.tile([grid_y[0], grid_y[-1]], len(grid_x)),
            color="k", lw=0.6, alpha=0.55)
    ax.plot(np.tile([grid_x[0], grid_x[-1]], len(grid_y)),
            np.repeat(grid_y, 2), color="k", lw=0.6, alpha=0.55)
    # 格点标记
    for gx in grid_x:
        for gy in grid_y:
            ax.plot(gx, gy, "o", ms=4, color="k", zorder=5)
    # 单元内一候选镜位（示意），及其到四角格点的连线
    p = base + np.array([0.55, 0.62]) * GRID_STEP
    ax.plot(*p, marker="s", ms=7, color="tab:green", zorder=7,
            markeredgecolor="k", markeredgewidth=0.5)
    for gx in (base[0], base[0] + GRID_STEP):
        for gy in (base[1], base[1] + GRID_STEP):
            ax.plot([p[0], gx], [p[1], gy], color="tab:green", lw=0.8,
                    ls="--", alpha=0.7, zorder=4)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("(b) 网格预计算与插值查表（局部放大）")
    ax.text(0.03, 0.97,
            "方形：候选镜位 $p$\n"
            "其贡献由所在网格单元\n"
            "四个格点 $\\Phi$ 值双线性插值\n"
            "获得，无需逐镜追迹",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999", lw=0.6))

    fig.suptitle("快速效率地图：平面网格预计算 $\\Phi(p)$，优化中直接插值查表",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fig.savefig(FIG_PATH, dpi=200)
    print(f"已保存: {FIG_PATH}")


if __name__ == "__main__":
    main()
