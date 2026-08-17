"""问题二优化求解器：六角晶格布局 + 贪心选镜 + 差分进化 + 两级精度复核。

优化模型（与论文 5.2 一致）：
    min  A_total = N·w·H
    s.t. P̄ ≥ 60 MW（60 个代表时刻等权平均口径）
         |p_i| ≤ 350 m（场半径），|p_i − T| ≥ 100 m（塔周禁装区）
         |p_i − p_j| ≥ w + 5 m（相邻底座间距）
         2 ≤ w,H ≤ 8 m，H ≤ w，2 ≤ a ≤ 6 m，a ≥ H/2

求解框架（论文 5.2）：
    外层：差分进化（DE）在 (T, w, H, a, θ, phase_x, phase_y) 8 维空间搜索；
    内层：六角晶格生成候选镜位 → 解析年均贡献排序 → 贪心选镜；
    两级精度：DE 内低精度 60 时刻追迹（6×6 网格、32 光线/点）；
             精英解高精度复核（10×10、1024 光线/点）。

本文件同时提供参数化追迹函数（塔位/镜像/安装高可配置），
问题三（q3_optimizer.py）直接复用。
"""

import os

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from q1_functions import (
    COLLECTOR_Z_HIGH,
    COLLECTOR_Z_LOW,
    COLLECTOR_RADIUS,
    ETA_REF,
    EXCLUSION_RADIUS,
    FIELD_RADIUS,
    REP_HOURS,
    SAMPLING_EPS,
    atmospheric_transmittance,
    cone_ray_directions,
    dni,
    mirror_basis,
    sun_geometry,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(ROOT, "02_论文", "figures")
OUTPUT_DIR = os.path.join(ROOT, "02_论文", "files")

# ============ 问题二参数 ============
RATED_POWER_MW = 60.0                 # 额定年平均输出热功率（MW）
TOWER_HEIGHT = 80.0                   # 吸收塔高度（集热器中心离地，m）
W_MIN, W_MAX = 2.0, 8.0               # 镜面宽度/高度范围（m）
A_MIN, A_MAX = 2.0, 6.0               # 安装高度范围（m）
LATTICE_MARGIN = 5.0                  # 相邻底座间距裕量：d = w + 5

FAST_GRID, FAST_RAYS = 6, 32          # 低精度（DE 内评价）
FINAL_GRID, FINAL_RAYS = 10, 1024     # 高精度（最终复核）

DE_POP, DE_MAXITER, DE_SEED = 8, 12, 2023
GREEDY_DISCOUNT = 0.75                # 贪心初选折扣（保守估计年均遮挡/截断）
LAMBDA = 1e4                          # 罚函数系数（1 MW 缺口罚 1e4 ≈ 总面积量级）
P_FINAL_TOL = 0.30                    # 复核阶段功率达标容差（MW）

# ============ 参数化追迹函数（塔位 / 镜像 / 安装高可配置） ============

def param_terms(mirror_center, tower_xy):
    """由镜心与塔位计算位置相关量（t、d_hr、eta_at），可缓存复用。"""
    collector = np.array([tower_xy[0], tower_xy[1], TOWER_HEIGHT])
    to_collector = collector - mirror_center
    d_hr = np.linalg.norm(to_collector, axis=1)
    t = to_collector / d_hr[:, None]
    eta_at = atmospheric_transmittance(d_hr)
    return t, d_hr, eta_at


def param_attitude(mirror_center, s, tower_xy, terms=None):
    """镜面姿态与解析效率（η_cos、η_at），塔位可配置。"""
    if terms is None:
        terms = param_terms(mirror_center, tower_xy)
    t, d_hr, eta_at = terms
    sum_vec = s[None, :] + t
    n = sum_vec / np.linalg.norm(sum_vec, axis=1)[:, None]
    eta_cos = np.sum(s[None, :] * n, axis=1)
    return t, n, eta_cos, d_hr, eta_at


def _sample_grid(mirror_center, n, w, H, grid_n):
    u, v = mirror_basis(n)
    frac = (np.arange(grid_n) + 0.5) / grid_n - 0.5
    du, dv = np.meshgrid(frac, frac)
    off = (du.ravel()[None, :, None] * w * u[:, None, :]
           + dv.ravel()[None, :, None] * H * v[:, None, :])
    return mirror_center[:, None, :] + off


def neighbor_radius(w):
    """候选遮挡镜筛选半径：R = √(d∥max² + d⊥max²) + margin。
    d⊥max = w√2（两镜半对角线之和），d∥max = 15.83（相邻镜间距上界）。"""
    return float(np.hypot(15.83, w * np.sqrt(2.0))) + 1.5


def param_neighbors(mirror_xy, radius):
    """按水平距离预计算每面镜的候选遮挡镜索引（不含自身）。"""
    from scipy.spatial import cKDTree

    tree = cKDTree(mirror_xy)
    pairs = tree.query_pairs(radius)
    neighbors = [[] for _ in range(len(mirror_xy))]
    for i, j in pairs:
        neighbors[i].append(j)
        neighbors[j].append(i)
    return [np.asarray(nb, dtype=int) for nb in neighbors]


def param_tower_shadow(points, s, tower_xy, radius=3.5):
    """塔身+集热器合并圆柱阴影判定（圆柱中心为塔位，z ∈ [0, 84]）。"""
    d = s
    A = d[0] ** 2 + d[1] ** 2
    hit = np.zeros(len(points), dtype=bool)
    if A <= 1e-12:
        return hit
    B = 2.0 * ((points[:, 0] - tower_xy[0]) * d[0]
               + (points[:, 1] - tower_xy[1]) * d[1])
    C = ((points[:, 0] - tower_xy[0]) ** 2
         + (points[:, 1] - tower_xy[1]) ** 2 - radius ** 2)
    disc = B ** 2 - 4.0 * A * C
    ok = disc >= 0
    lam = (-B - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * A)
    z_hit = points[:, 2] + lam * d[2]
    return ok & (lam > SAMPLING_EPS) & (z_hit >= 0.0) & (z_hit <= COLLECTOR_Z_HIGH)


def _ray_rect_batch(points, d, C, n, u, v, half_w, half_h):
    """批量判定采样点光线是否被候选镜面中任一面遮挡（矩形半宽/半高可配置）。"""
    rel = C[None, :, :] - points[:, None, :]
    denom = d @ n.T
    lam = np.sum(rel * n[None, :, :], axis=-1) / denom[None, :]
    Q = points[:, None, :] + lam[..., None] * d
    in_rect = ((np.abs(np.sum((Q - C[None, :, :]) * u[None, :, :], axis=-1)) <= half_w)
               & (np.abs(np.sum((Q - C[None, :, :]) * v[None, :, :], axis=-1)) <= half_h))
    hit = (np.abs(denom) > 1e-9) & (lam > SAMPLING_EPS) & in_rect
    return hit.any(axis=1)


def _ray_rect_batch_vec(P, d, C, n, u, v, half_w, half_h, mask):
    """向量化阴影/挡光判定：(m, S, K, 3) 批量，mask 屏蔽无效候选（pad 位）。"""
    rel = C[:, None, :, :] - P[:, :, None, :]          # (m, S, K, 3)
    denom = np.sum(d[:, None, :] * n, axis=-1)         # (m, K)
    lam = np.sum(rel * n[:, None, :, :], axis=-1) / denom[:, None, :]  # (m, S, K)
    Q = P[:, :, None, :] + lam[..., None] * d[:, None, None, :]
    in_rect = ((np.abs(np.sum((Q - C[:, None, :, :]) * u[:, None, :, :], axis=-1)) <= half_w)
               & (np.abs(np.sum((Q - C[:, None, :, :]) * v[:, None, :, :], axis=-1)) <= half_h))
    hit = ((np.abs(denom) > 1e-9)[:, None, :] & (lam > SAMPLING_EPS)
           & in_rect & mask[:, None, :])
    return hit.any(axis=-1)                            # (m, S)


def param_effective_points(mirror_center, s, n, neighbors, w, H,
                           tower_xy, grid_n, chunk=300):
    """采样点 + 有效掩码（塔阴影 → 镜间阴影 → 镜间挡光），分块向量化。"""
    P = _sample_grid(mirror_center, n, w, H, grid_n)
    S = P.shape[1]
    tower_hit = param_tower_shadow(P.reshape(-1, 3), s, tower_xy).reshape(-1, S)

    r = 2.0 * np.sum(s * n, axis=1)[:, None] * n - s
    u, v = mirror_basis(n)
    half_w, half_h = w / 2.0, H / 2.0
    valid = ~tower_hit

    N = len(mirror_center)
    k_max = max((len(nb) for nb in neighbors), default=0)
    if k_max == 0:
        return P, valid
    pad = np.zeros((N, k_max), dtype=int)
    mask = np.zeros((N, k_max), dtype=bool)
    for i, nb in enumerate(neighbors):
        if len(nb):
            pad[i, :len(nb)] = nb
            mask[i, :len(nb)] = True

    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        idx = pad[lo:hi]
        msk = mask[lo:hi]
        C = mirror_center[idx]
        nC = n[idx]
        uC = u[idx]
        vC = v[idx]
        Pc = P[lo:hi]
        s_b = np.tile(s, (hi - lo, 1))
        valid[lo:hi] &= ~_ray_rect_batch_vec(Pc, s_b, C, nC, uC, vC,
                                             half_w, half_h, msk)
        valid[lo:hi] &= ~_ray_rect_batch_vec(Pc, r[lo:hi], C, nC, uC, vC,
                                             half_w, half_h, msk)
    return P, valid


def param_trunc(P, s, n, valid, tower_xy, n_rays, chunk=300):
    """每面镜截断效率（光锥 Sobol 采样 + 圆柱求交），分块向量化。"""
    dirs = cone_ray_directions(s, n_rays)
    cx, cy = tower_xy[0], tower_xy[1]
    r2_cyl = COLLECTOR_RADIUS ** 2
    S = P.shape[1]
    out = np.zeros(P.shape[0])

    for lo in range(0, P.shape[0], chunk):
        Pc = P[lo:lo + chunk]                 # (m, S, 3)
        nc = n[lo:lo + chunk]                 # (m, 3)
        vc = valid[lo:lo + chunk]             # (m, S)
        n_valid = vc.sum(axis=1)
        if n_valid.sum() == 0:
            continue

        cos_mk = nc @ dirs.T                  # (m, K)
        r = 2.0 * cos_mk[:, :, None] * nc[:, None, :] - dirs[None, :, :]  # (m, K, 3)
        A = r[:, :, 0] ** 2 + r[:, :, 1] ** 2                    # (m, K)
        A_ok = A > 1e-12
        B = 2.0 * ((Pc[:, :, None, 0] - cx) * r[:, None, :, 0]
                   + (Pc[:, :, None, 1] - cy) * r[:, None, :, 1])  # (m, S, K)
        C = (Pc[:, :, 0] - cx) ** 2 + (Pc[:, :, 1] - cy) ** 2 - r2_cyl  # (m, S)
        disc = B ** 2 - 4.0 * A[:, None, :] * C[:, :, None]
        lam = (-B - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * A[:, None, :])
        z_hit = Pc[:, :, 2:3] + lam * r[:, None, :, 2]
        hit = (A_ok[:, None, :] & (disc >= 0) & (lam > SAMPLING_EPS)
               & (z_hit >= COLLECTOR_Z_LOW) & (z_hit <= COLLECTOR_Z_HIGH))
        hit &= vc[:, :, None]
        out[lo:lo + chunk] = hit.sum(axis=(1, 2)) / np.maximum(n_valid * n_rays, 1)
    return out


# ============ 布局生成与快速评价 ============

def hexagonal_lattice(tower_xy, d, theta, phase):
    """六角晶格候选镜位（间距 d，旋转角 θ，胞内相位 phase）。

    基向量 a1 = d(cosθ, sinθ)，a2 = d(cos(θ+60°), sin(θ+60°))。
    裁剪：|p| ≤ 350（场地圆），|p − T| ≥ 100（塔周禁装区）。
    """
    a1 = d * np.array([np.cos(theta), np.sin(theta)])
    a2 = d * np.array([np.cos(theta + np.pi / 3.0), np.sin(theta + np.pi / 3.0)])
    n_span = int(np.ceil((FIELD_RADIUS + d * 2) / d)) + 2
    m = np.arange(-n_span, n_span + 1)
    n = np.arange(-n_span, n_span + 1)
    M, Nn = np.meshgrid(m, n)
    P = (tower_xy + M.ravel()[:, None] * a1 + Nn.ravel()[:, None] * a2
         + np.asarray(phase)[None, :])
    r0 = np.hypot(P[:, 0], P[:, 1])
    rt = np.hypot(P[:, 0] - tower_xy[0], P[:, 1] - tower_xy[1])
    mask = (r0 <= FIELD_RADIUS) & (rt >= EXCLUSION_RADIUS)
    return P[mask]


def analytic_contribution(p_xy, tower_xy, a):
    """解析年均单位面积功率上限 c_i = mean_t [DNI_t·η_cos(t)·η_at]（kW/m²）。

    无阴影/截断近似，仅用于贪心排序；真实功率由追迹复核。
    """
    mc = np.column_stack([p_xy, np.full(len(p_xy), a)])
    t, d_hr, eta_at = param_terms(mc, tower_xy)
    acc = np.zeros(len(p_xy))
    for month in range(1, 13):
        for hour in REP_HOURS:
            s = sun_geometry(month, hour)[4]
            sum_vec = s[None, :] + t
            n = sum_vec / np.linalg.norm(sum_vec, axis=1)[:, None]
            eta_cos = np.sum(s[None, :] * n, axis=1)
            acc += dni(s[2]) * eta_cos * eta_at
    return acc / 60.0


def greedy_select(p_xy, tower_xy, w, H, a):
    """贪心选镜：按解析贡献降序，前缀面积功率达目标即止。

    目标 = 额定功率 / 预期总效率折扣（η_ref × η_sb̄ × η_trunc̄ ≈ 0.80），
    即按"无损失贡献"多选 ~25% 裕量，由后续精确追迹与补镜校正。
    """
    c = analytic_contribution(p_xy, tower_xy, a)
    order = np.argsort(-c)
    target_kw = (RATED_POWER_MW * 1000.0
                 / (ETA_REF * GREEDY_DISCOUNT))
    csum = np.cumsum(c[order]) * w * H
    K = int(np.searchsorted(csum, target_kw, side="left")) + 1
    K = min(K, len(p_xy))
    return p_xy[order[:K]], order, c


def field_eval(sel_xy, tower_xy, w, H, a, grid_n, n_rays, verbose=False):
    """60 个代表时刻完整追迹评价，返回 (P̄_MW, 分项年均效率 dict)。

    口径与问题一一致：12 个月 21 日 × 5 时刻等权平均。
    """
    if len(sel_xy) == 0:
        return 0.0, {}
    mc = np.column_stack([sel_xy, np.full(len(sel_xy), a)])
    area = w * H
    radius = neighbor_radius(w)
    neighbors = param_neighbors(sel_xy, radius)
    terms = param_terms(mc, tower_xy)

    acc_p, acc_cos, acc_sb, acc_trunc, acc_eta = 0.0, 0.0, 0.0, 0.0, 0.0
    for month in range(1, 13):
        for hour in REP_HOURS:
            s = sun_geometry(month, hour)[4]
            _, n, eta_cos, _, eta_at = param_attitude(mc, s, tower_xy, terms)
            P, valid = param_effective_points(mc, s, n, neighbors, w, H,
                                              tower_xy, grid_n)
            eta_sb = valid.mean(axis=1)
            eta_trunc = param_trunc(P, s, n, valid, tower_xy, n_rays)
            eta_i = ETA_REF * eta_cos * eta_at * eta_sb * eta_trunc
            p = dni(s[2]) * area * eta_i.sum() / 1000.0
            acc_p += p
            acc_cos += np.mean(eta_cos)
            acc_sb += np.mean(eta_sb)
            acc_trunc += np.mean(eta_trunc)
            acc_eta += np.sum(eta_i) / len(eta_i)
    n_t = 60
    return (acc_p / n_t,
            {"eta_cos": acc_cos / n_t, "eta_sb": acc_sb / n_t,
             "eta_trunc": acc_trunc / n_t, "eta": acc_eta / n_t})


def fitness(x, grid_n=FAST_GRID, n_rays=FAST_RAYS):
    """DE 适应度：低精度 60 时刻追迹功率 + 总面积，罚函数处理功率缺口。"""
    tx, ty, w, H, a, theta, phx, phy = x
    w = float(np.clip(w, W_MIN, W_MAX))
    H = float(np.clip(min(H, w), W_MIN, W_MAX))
    a = float(np.clip(a, A_MIN, A_MAX))
    a = max(a, H / 2.0)
    d = w + LATTICE_MARGIN
    tower = np.array([tx, ty])

    cand = hexagonal_lattice(tower, d, theta, (phx % d, phy % d))
    if len(cand) < 200:
        return 1e12

    sel, _, _ = greedy_select(cand, tower, w, H, a)
    if len(sel) == 0:
        return 1e12

    P_est, _ = field_eval(sel, tower, w, H, a, grid_n, n_rays)
    area = len(sel) * w * H
    gap = max(0.0, RATED_POWER_MW - P_est)
    return float(area + LAMBDA * gap ** 2)


def run_de(popsize=DE_POP, maxiter=DE_MAXITER, seed=DE_SEED,
           log_path=None, workers=1):
    """差分进化搜索 8 维决策空间；返回 (result, 每代最优个体列表)。

    workers>1 时多进程并行评价（scipy 内部实现），进程数不超过机器核数。
    """
    bounds = [(-320.0, 320.0), (-320.0, 320.0),      # 塔位 (T_x, T_y)
              (W_MIN, W_MAX), (W_MIN, W_MAX),        # 镜宽 w、镜高 H
              (A_MIN, A_MAX),                        # 安装高度 a
              (0.0, np.pi / 3.0),                    # 晶格旋转角 θ
              (0.0, 13.0), (0.0, 13.0)]              # 相位 (phase_x, phase_y)

    history, best_per_gen = [], []

    def callback(xk, convergence):
        F = fitness(xk)
        history.append((xk.copy(), F, convergence))
        best_per_gen.append(xk.copy())
        if log_path and len(history) % 20 == 0:
            np.save(log_path, np.array(history, dtype=object))

    result = differential_evolution(
        fitness, bounds, popsize=popsize, maxiter=maxiter, seed=seed,
        strategy="best1bin", mutation=(0.5, 1.0), recombination=0.9,
        callback=callback, tol=1e-10, polish=True, workers=workers)
    return result, best_per_gen


def final_review(best_x, grid_n=FINAL_GRID, n_rays=FINAL_RAYS,
                 max_adjust=4):
    """最优解的复核与局部调整（高精度 60 时刻追迹）。

    功率不足 → 按解析贡献补镜；超标过多 → 删最低贡献镜。
    """
    tx, ty, w, H, a, theta, phx, phy = best_x
    w = float(np.clip(w, W_MIN, W_MAX))
    H = float(np.clip(min(H, w), W_MIN, W_MAX))
    a = float(np.clip(a, A_MIN, A_MAX))
    a = max(a, H / 2.0)
    d = w + LATTICE_MARGIN
    tower = np.array([tx, ty])
    cand = hexagonal_lattice(tower, d, theta, (phx % d, phy % d))
    sel, order, c = greedy_select(cand, tower, w, H, a)

    picked = set(order[:len(sel)].tolist())
    tail = [i for i in order if i not in picked]
    for _ in range(max_adjust):
        P_est, eff = field_eval(sel, tower, w, H, a, grid_n, n_rays)
        if P_est < RATED_POWER_MW - P_FINAL_TOL and tail:
            nxt = tail.pop(0)
            picked.add(nxt)
            sel = np.vstack([sel, cand[nxt]])
        elif P_est > RATED_POWER_MW + 1.0 and len(sel) > 100:
            # 删贡献最小（最后选入）的镜面
            drop = order[len(sel) - 1]
            picked.discard(drop)
            keep = [i for i in order if i in picked]
            sel = cand[keep]
        else:
            break
    P_final, eff = field_eval(sel, tower, w, H, a, grid_n, n_rays)
    return sel, tower, w, H, a, P_final, eff


def export_result2(sel_xy, tower, w, H, a, P_mw, path=None):
    """按模板格式导出 result2.xlsx（塔位 / 每镜序号 / 尺寸 / 坐标）。"""
    if path is None:
        path = os.path.join(ROOT, "00_题目与数据", "result2.xlsx")
    df = pd.DataFrame({
        "吸收塔x坐标 (m)": float(tower[0]),
        "吸收塔y坐标 (m)": float(tower[1]),
        "定日镜序号": np.arange(1, len(sel_xy) + 1),
        "定日镜宽度 (m)": float(w),
        "定日镜高度 (m)": float(H),
        "定日镜x坐标 (m)": sel_xy[:, 0],
        "定日镜y坐标 (m)": sel_xy[:, 1],
        "定日镜z坐标 (m)": float(a),
    })
    df.to_excel(path, index=False)
    return path


def plot_q2_layout(sel_xy, tower, save_path=None):
    """镜场布局俯视图（镜位 + 塔位 + 场地/禁装区圆）。"""
    import matplotlib.pyplot as plt

    from q1_functions import _set_cjk_font

    _set_cjk_font()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(sel_xy[:, 0], sel_xy[:, 1], s=0.8, c="tab:blue", label="定日镜")
    ax.scatter(*tower, c="tab:red", marker="*", s=160, label="吸收塔")
    ax.add_patch(plt.Circle((0, 0), FIELD_RADIUS, fill=False, ls="--", color="k"))
    ax.add_patch(plt.Circle(tower, EXCLUSION_RADIUS, fill=False, ls=":",
                            color="gray"))
    ax.set_aspect("equal")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title(f"问题二优化布局（N={len(sel_xy)}，塔位 ({tower[0]:.1f}, {tower[1]:.1f})）")
    ax.legend(loc="upper right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def main():
    print("=" * 60)
    print("问题二：定日镜场优化设计（六角晶格 + 贪心选镜 + 差分进化）")
    print("=" * 60)

    result, best_per_gen = run_de(workers=8)
    print(f"\nDE 完成: 迭代 {DE_MAXITER} 代, 最优适应度 F = {result.fun:.1f}")
    print(f"  DE 最优: T = ({result.x[0]:.2f}, {result.x[1]:.2f}) m, "
          f"镜面 {result.x[2]:.2f}×{result.x[3]:.2f} m, "
          f"安装高 {result.x[4]:.2f} m, θ={result.x[5]:.3f}")

    # 精英筛选：每代最优个体 + DE 最终解，用低精度 60 时刻精确追迹排序
    candidates = list(best_per_gen)
    candidates.append(result.x)
    scored = []
    for xk in candidates:
        tx, ty, w, H, a, theta, phx, phy = xk
        w = float(np.clip(w, W_MIN, W_MAX))
        H = float(np.clip(min(H, w), W_MIN, W_MAX))
        a = float(np.clip(a, A_MIN, A_MAX))
        a = max(a, H / 2.0)
        tower = np.array([tx, ty])
        d = w + LATTICE_MARGIN
        cand = hexagonal_lattice(tower, d, theta, (phx % d, phy % d))
        sel, _, _ = greedy_select(cand, tower, w, H, a)
        P_est, _ = field_eval(sel, tower, w, H, a, FAST_GRID, FAST_RAYS)
        scored.append((xk, sel, tower, w, H, a, P_est, len(sel) * w * H))
        print(f"  候选: 塔({tx:7.1f},{ty:7.1f}) 镜{w:4.2f}×{H:4.2f} "
              f"a={a:4.2f} N={len(sel):5d} P={P_est:6.2f}MW")

    scored.sort(key=lambda t: (t[6] < RATED_POWER_MW, t[7]))  # 先达标，再小面积

    # 高精度复核（按低精度排序取前 3，逐一下探到达标）
    chosen = None
    for xk, sel, tower, w, H, a, P_low, A_low in scored[:3]:
        print(f"\n高精度复核（60 时刻，10×10 网格，1024 光线）: "
              f"塔({tower[0]:.1f},{tower[1]:.1f}) ...")
        sel_f, tower_f, w_f, H_f, a_f, P_final, eff = final_review(
            xk, grid_n=FINAL_GRID, n_rays=FINAL_RAYS)
        if P_final >= RATED_POWER_MW - P_FINAL_TOL:
            chosen = (sel_f, tower_f, w_f, H_f, a_f, P_final, eff)
            break
        print(f"  → 复核 P={P_final:.2f}MW 未达标，尝试下一候选")

    if chosen is None:
        raise RuntimeError("所有精英复核均未达到 60MW，需增加 DE 迭代或放宽约束")

    sel, tower, w, H, a, P_final, eff = chosen
    A_total = len(sel) * w * H
    q = P_final * 1000.0 / A_total
    print("\n======== 问题二最终结果 ========")
    print(f"  吸收塔位置: ({tower[0]:.1f}, {tower[1]:.1f}) m")
    print(f"  镜面尺寸: {w:.2f} × {H:.2f} m², 安装高度: {a:.2f} m")
    print(f"  定日镜数: {len(sel)}, 镜面总面积: {A_total:.1f} m²")
    print(f"  年平均输出热功率: {P_final:.3f} MW")
    print(f"  单位镜面面积年平均输出热功率: {q:.4f} kW/m²")
    print(f"  年均分项效率: 余弦 {eff['eta_cos']:.4f}, "
          f"阴影遮挡 {eff['eta_sb']:.4f}, 截断 {eff['eta_trunc']:.4f}, "
          f"综合 {eff['eta']:.4f}")

    path = export_result2(sel, tower, w, H, a, P_final)
    print(f"  result2.xlsx → {path}")
    fig = plot_q2_layout(sel, tower,
                         os.path.join(FIGURE_DIR, "fig_q2_layout.png"))
    print(f"  布局图 → {fig}")

    summary = {
        "tower": tower, "w": w, "H": H, "a": a, "N": len(sel),
        "A_total": A_total, "P_mw": P_final, "q": q, **eff,
    }
    np.save(os.path.join(OUTPUT_DIR, "q2_result.npy"), summary)
    print("  结果已缓存 → 02_论文/files/q2_result.npy")
    return summary


if __name__ == "__main__":
    main()
