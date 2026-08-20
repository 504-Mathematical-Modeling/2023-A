"""生成问题二/三布局补充图（论文 5.2/5.3 结果小节）。

- fig_q2_layout.png     问题二优化镜场布局（等尺寸，六角晶格 + 贪心选镜）
- fig_q3_layout.png     问题三异尺寸镜场布局（每镜按"单位面积贡献"选档，
                        镜宽以渐变色显示 + colorbar）
- fig_q3_size_dist.png  问题三常见尺寸档位占比条形图

布局形态与论文结果小节描述一致（塔位/镜面尺寸/晶格间距），
图仅作形态示意，不标注镜数。
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from q1_functions import _set_cjk_font, sun_geometry, REP_HOURS
from q2_optimizer import (hexagonal_lattice, greedy_select, analytic_contribution,
                          param_terms, param_attitude, _sample_grid, param_trunc,
                          param_effective_points, param_neighbors,
                          FIELD_RADIUS, EXCLUSION_RADIUS)

FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "02_论文", "figures")
plt.rcParams["axes.unicode_minus"] = False


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_aspect("equal")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")


def _plot_field_frame(ax, tower, radius_100=True):
    ax.add_patch(Circle((0, 0), FIELD_RADIUS, fill=False,
                        edgecolor="#999999", linestyle="-", linewidth=0.9))
    if radius_100:
        ax.add_patch(Circle(tuple(tower), EXCLUSION_RADIUS, fill=False,
                            edgecolor="#999999", linestyle="--", linewidth=0.9))
    ax.plot(tower[0], tower[1], marker="*", ms=13, color="#C00000",
            mec="white", mew=0.8, zorder=5, label="吸收塔")
    ax.annotate("吸收塔", (tower[0], tower[1]), textcoords="offset points",
                xytext=(10, -14), fontsize=10)
    ax.annotate("场地边界 r=350 m", (0, FIELD_RADIUS + 8),
                ha="center", fontsize=8.5, color="#555555")
    ax.set_xlim(-360, 360)
    ax.set_ylim(-360, 360)


def plot_q2_layout(tower=(52.4, -118.6), w=5.8, H=5.8, a=2.9,
                   theta=0.0, save_path=None):
    d = w + 5.0
    cand = hexagonal_lattice(np.array(tower), d, theta, (0.0, 0.0))
    sel, _, _ = greedy_select(cand, np.array(tower), w, H, a)

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    _plot_field_frame(ax, tower)
    ax.scatter(sel[:, 0], sel[:, 1], s=2.2, color="#2F5597",
               linewidths=0, zorder=2)
    ax.annotate(f"六角晶格间距 d = w + 5 = {d:.1f} m",
                (0, -FIELD_RADIUS - 26), ha="center", fontsize=9,
                color="#333333")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    _style_ax(ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def _eta_sb_trunc_mean(p_xy, w, H, a, tower, neighbors, n_rays=64):
    """60 时刻平均的阴影遮挡与截断效率（无塔影，4×4 采样网格，选档依据）。"""
    mc = np.column_stack([p_xy, np.full(len(p_xy), a)])
    terms = param_terms(mc, tower)
    acc_sb = np.zeros(len(p_xy))
    acc_t = np.zeros(len(p_xy))
    for month in range(1, 13):
        for hour in REP_HOURS:
            s = sun_geometry(month, hour)[4]
            _, n, _, _, _ = param_attitude(mc, s, tower, terms)
            P, valid = param_effective_points(mc, s, n, neighbors,
                                              w, H, tower, 4)
            acc_sb += valid.sum(axis=1) / valid.shape[1]
            acc_t += param_trunc(P, s, n, valid, tower, n_rays)
    return acc_sb / 60.0, acc_t / 60.0


def q3_choose_levels(p_xy, tower):
    """逐镜档位选择：枚举 (w,H)（1 m 步长，2≤H≤w≤8），a=H/2，
    按"贡献" Φ = c·η_sb·η_trunc·w·H 选最优档位：c 与尺寸无关，
    截断偏爱小镜、阴影遮挡惩罚小镜、镜面面积偏爱大镜，三者权衡
    形成"近塔小镜、远塔大镜"的尺寸梯度（示意图用，完整流程以
    复核层追迹为准）。近邻半径取 25 m 固定，所有档位共用。"""
    neighbors = param_neighbors(p_xy, 25.0)
    best_w = np.zeros(len(p_xy))
    best_H = np.zeros(len(p_xy))
    best_phi = np.full(len(p_xy), -np.inf)
    for w in np.arange(2.0, 8.0 + 1e-9, 1.0):
        for H in np.arange(2.0, w + 1e-9, 1.0):
            a = max(H / 2.0, 2.0)
            c = analytic_contribution(p_xy, tower, a)
            eta_sb, eta_t = _eta_sb_trunc_mean(p_xy, w, H, a, tower,
                                               neighbors)
            phi = c * eta_sb * eta_t * w * H
            better = phi > best_phi
            best_w[better] = w
            best_H[better] = H
            best_phi[better] = phi[better]
    return best_w, best_H


def plot_q3_layout(tower=(64.8, -110.2), d=13.0, theta=0.0, save_path=None):
    cand = hexagonal_lattice(np.array(tower), d, theta, (0.0, 0.0))
    w_i, h_i = q3_choose_levels(cand, tower)

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    _plot_field_frame(ax, tower)
    sc = ax.scatter(cand[:, 0], cand[:, 1], c=w_i, s=2.6,
                    cmap="viridis", vmin=2.0, vmax=8.0,
                    linewidths=0, zorder=2)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("镜宽 $w$ / m", fontsize=9.5)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    _style_ax(ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax, w_i, h_i


def plot_q3_size_dist(w_i, h_i, save_path=None):
    """问题三常见尺寸档位占比（按 w×H 组合计数，取前 12）。"""
    import pandas as pd
    df = pd.DataFrame({"w": w_i, "H": h_i})
    counts = df.value_counts().head(12).sort_values()
    labels = [f"{w:.1f}×{h:.1f}" for w, h in counts.index]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.barh(labels, counts.values, color="#A6C4E8",
            edgecolor="#2F5597", linewidth=0.7, height=0.62)
    for i, v in enumerate(counts.values):
        ax.text(v + 4, i, str(v), va="center", fontsize=8.5, color="#333333")
    ax.set_xlabel("镜面数")
    ax.set_title("问题三常见尺寸档位（镜宽×镜高 / m²）", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def main():
    _set_cjk_font()
    os.makedirs(FIGURE_DIR, exist_ok=True)

    fig2, _ = plot_q2_layout(
        save_path=os.path.join(FIGURE_DIR, "fig_q2_layout.png"))
    print("fig_q2_layout.png 生成")

    fig3, _, w_i, h_i = plot_q3_layout(
        save_path=os.path.join(FIGURE_DIR, "fig_q3_layout.png"))
    print(f"fig_q3_layout.png 生成（{len(w_i)} 面镜，"
          f"w 分布 {w_i.min():.1f}–{w_i.max():.1f} m）")

    figd, _ = plot_q3_size_dist(
        w_i, h_i, save_path=os.path.join(FIGURE_DIR, "fig_q3_size_dist.png"))
    print("fig_q3_size_dist.png 生成")

    np.savez(os.path.join(FIGURE_DIR, "..", "files", "q23_layout_demo.npz"),
             q2_sel=None, q3_xy=None, q3_w=w_i, q3_h=h_i)


if __name__ == "__main__":
    main()