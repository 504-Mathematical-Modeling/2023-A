"""通用建模组件：参数化追迹评价、六角晶格布局、贪心选镜与公共常量。

问题一/二/三共用的物理常量、太阳追迹与效率计算函数、布局生成与
快速评价接口集中于此，供 q2_optimizer / q3_optimizer / refine_* /
verify_* 等脚本复用。

口径说明（与论文一致）：
    - 年均效率按 12 个月 21 日 × 5 个代表时刻（9:00--15:00）等权平均；
    - 单镜光学效率 η_i = η_ref · η_cos · η_at · η_sb · η_trunc；
    - 追迹精度分三档：低精度 6×6 网格 / 32 光线（DE 内），
      复筛 8×8 / 512 光线，最终报告 10×10 / 1024 光线。
"""

import os

import numpy as np
import pandas as pd

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

# ============ 公共参数 ============
RATED_POWER_MW = 60.0                 # 额定年平均输出热功率（MW）
TOWER_HEIGHT = 80.0                   # 吸收塔高度（集热器中心离地，m）
W_MIN, W_MAX = 2.0, 8.0               # 镜面宽度/高度范围（m）
A_MIN, A_MAX = 2.0, 6.0               # 安装高度范围（m）
LATTICE_MARGIN = 5.0                  # 相邻底座间距裕量：d = w + 5

FAST_GRID, FAST_RAYS = 6, 32          # 低精度（DE 内评价）
FINAL_GRID, FINAL_RAYS = 10, 1024     # 高精度（最终复核）

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
    """镜面采样网格；w/H 可为标量（全场统一）或逐镜向量。"""
    u, v = mirror_basis(n)
    frac = (np.arange(grid_n) + 0.5) / grid_n - 0.5
    du, dv = np.meshgrid(frac, frac)
    wv = np.broadcast_to(np.asarray(w, float),
                         (len(mirror_center),))[:, None, None]
    Hv = np.broadcast_to(np.asarray(H, float),
                         (len(mirror_center),))[:, None, None]
    off = (du.ravel()[None, :, None] * wv * u[:, None, :]
           + dv.ravel()[None, :, None] * Hv * v[:, None, :])
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
    """向量化阴影/挡光判定：(m, S, K, 3) 批量，mask 屏蔽无效候选（pad 位）。

    half_w/half_h 可为标量（全场统一）或 (m, K) 数组（逐候选镜半宽/半高）。
    """
    rel = C[:, None, :, :] - P[:, :, None, :]          # (m, S, K, 3)
    denom = np.sum(d[:, None, :] * n, axis=-1)         # (m, K)
    lam = np.sum(rel * n[:, None, :, :], axis=-1) / denom[:, None, :]  # (m, S, K)
    Q = P[:, :, None, :] + lam[..., None] * d[:, None, None, :]
    hw = np.asarray(half_w)
    hh = np.asarray(half_h)
    if hw.ndim == 0:
        in_rect = ((np.abs(np.sum((Q - C[:, None, :, :]) * u[:, None, :, :],
                                  axis=-1)) <= hw)
                   & (np.abs(np.sum((Q - C[:, None, :, :]) * v[:, None, :, :],
                                    axis=-1)) <= hh))
    else:
        in_rect = ((np.abs(np.sum((Q - C[:, None, :, :]) * u[:, None, :, :],
                                  axis=-1)) <= hw[:, None, :])
                   & (np.abs(np.sum((Q - C[:, None, :, :]) * v[:, None, :, :],
                                    axis=-1)) <= hh[:, None, :]))
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

    hw = np.asarray(w, float) / 2.0
    hh = np.asarray(H, float) / 2.0
    if hw.ndim == 0:
        half_w, half_h = hw, hh
    else:
        half_w = hw[pad]
        half_h = hh[pad]

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
        if hw.ndim == 0:
            half_wc, half_hc = half_w, half_h
        else:
            half_wc, half_hc = half_w[lo:hi], half_h[lo:hi]
        valid[lo:hi] &= ~_ray_rect_batch_vec(Pc, s_b, C, nC, uC, vC,
                                             half_wc, half_hc, msk)
        valid[lo:hi] &= ~_ray_rect_batch_vec(Pc, r[lo:hi], C, nC, uC, vC,
                                             half_wc, half_hc, msk)
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
    晶格补丁以场地中心 (0,0) 为原点覆盖整个场圆（塔位偏移不损失候选点），
    裁剪：|p| ≤ 350（场地圆），|p − T| ≥ 100（塔周禁装区，塔位仅用于此过滤）。
    """
    a1 = d * np.array([np.cos(theta), np.sin(theta)])
    a2 = d * np.array([np.cos(theta + np.pi / 3.0), np.sin(theta + np.pi / 3.0)])
    n_span = int(np.ceil((FIELD_RADIUS + d * 2) / d)) + 2
    m = np.arange(-n_span, n_span + 1)
    n = np.arange(-n_span, n_span + 1)
    M, Nn = np.meshgrid(m, n)
    P = (M.ravel()[:, None] * a1 + Nn.ravel()[:, None] * a2
         + np.asarray(phase)[None, :])
    r0 = np.hypot(P[:, 0], P[:, 1])
    rt = np.hypot(P[:, 0] - tower_xy[0], P[:, 1] - tower_xy[1])
    mask = (r0 <= FIELD_RADIUS) & (rt >= EXCLUSION_RADIUS)
    return P[mask]


def analytic_contribution(p_xy, tower_xy, a):
    """解析年均单位面积功率上限 c_i = mean_t [DNI_t·η_cos(t)·η_at]（kW/m²）。

    无阴影/截断近似，仅用于贪心排序；真实功率由追迹复核。
    """
    mc = np.column_stack([p_xy, np.broadcast_to(
        np.asarray(a, float), (len(p_xy),))])
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


def field_eval(sel_xy, tower_xy, w, H, a, grid_n, n_rays, verbose=False,
               return_per_mirror=False):
    """60 个代表时刻完整追迹评价，返回 (P̄_MW, 分项年均效率 dict)。

    口径与问题一一致：12 个月 21 日 × 5 时刻等权平均。
    w/H/a 可为标量（全场统一）或逐镜向量（长度 = 镜数，问题三分区方案）。
    return_per_mirror=True 时额外返回每镜年均功率数组 (N,) kW。
    """
    if len(sel_xy) == 0:
        return 0.0, {}
    a_sc = np.asarray(a, float)
    w_sc = np.asarray(w, float)
    H_sc = np.asarray(H, float)
    a_arr = np.broadcast_to(a_sc, (len(sel_xy),))
    w_arr = np.broadcast_to(w_sc, (len(sel_xy),))
    H_arr = np.broadcast_to(H_sc, (len(sel_xy),))
    mc = np.column_stack([sel_xy, a_arr])
    area = w_arr * H_arr
    radius = neighbor_radius(float(np.max(w_arr)))
    neighbors = param_neighbors(sel_xy, radius)
    terms = param_terms(mc, tower_xy)

    acc_p, acc_cos, acc_sb, acc_trunc, acc_eta = 0.0, 0.0, 0.0, 0.0, 0.0
    acc_pm = np.zeros(len(sel_xy)) if return_per_mirror else None
    for month in range(1, 13):
        for hour in REP_HOURS:
            s = sun_geometry(month, hour)[4]
            _, n, eta_cos, _, eta_at = param_attitude(mc, s, tower_xy, terms)
            w_eff = w_sc if w_sc.ndim == 0 else w_arr
            H_eff = H_sc if H_sc.ndim == 0 else H_arr
            P, valid = param_effective_points(mc, s, n, neighbors, w_eff, H_eff,
                                              tower_xy, grid_n)
            eta_sb = valid.mean(axis=1)
            eta_trunc = param_trunc(P, s, n, valid, tower_xy, n_rays)
            eta_i = ETA_REF * eta_cos * eta_at * eta_sb * eta_trunc
            p = dni(s[2]) * np.sum(area * eta_i) / 1000.0
            acc_p += p
            if return_per_mirror:
                acc_pm += dni(s[2]) * area * eta_i
            acc_cos += np.mean(eta_cos)
            acc_sb += np.mean(eta_sb)
            acc_trunc += np.mean(eta_trunc)
            acc_eta += np.sum(eta_i) / len(eta_i)
    n_t = 60
    res = (acc_p / n_t,
           {"eta_cos": acc_cos / n_t, "eta_sb": acc_sb / n_t,
            "eta_trunc": acc_trunc / n_t, "eta": acc_eta / n_t})
    if return_per_mirror:
        return res[0], res[1], acc_pm / n_t
    return res