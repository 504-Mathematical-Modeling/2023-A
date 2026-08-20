"""效率地图（思路 A）：单镜可分解效率的预计算与查表。

关键认识：
- η_cos、η_at 只依赖镜位置（解析式，无需追迹）；
- η_trunc 只依赖（镜位置, 时刻），与镜间布局完全无关；
- 因此可在镜场平面网格上一次性预计算，优化迭代中直接插值查表，
  将每镜 100×128=12800 条光线的追迹降为 O(1) 查表。

优化目标（额定年平均功率，60 时刻等权口径）：
  P̄ = η_ref · Σ_i A_i · η_sb_i · Φ(pos_i)
  其中 Φ(pos) = mean_t [ DNI(t)·η_cos(pos,t)·η_at(pos)·η_trunc(pos,t) ]
  （η_sb 依赖镜间布局，不进地图，由优化阶段单独处理/近似）

npz 输出字段：
- xs, ys         : 网格坐标（m），间距 MAP_GRID_STEP
- eta_cos_yr     : 年平均余弦效率 (ny, nx)
- eta_trunc_yr   : 年平均截断效率（无阴影假设口径）
- eta_at         : 大气透射率（不随时刻变化）
- phi            : 能量加权年平均效率（含 DNI 加权与 η_at）
- eta_trunc_t    : 逐时刻截断效率 (60, ny, nx)，供月平均分析
- eta_cos_t      : 逐时刻余弦效率 (60, ny, nx)
- dni_t          : 60 时刻 DNI (60,)
- months, hours  : 时刻对应的月/时
"""

import numpy as np
from q1_functions import (
    COLLECTOR_CENTER,
    COLLECTOR_RADIUS,
    COLLECTOR_Z_HIGH,
    COLLECTOR_Z_LOW,
    EXCLUSION_RADIUS,
    FIELD_RADIUS,
    GRID_N_DEFAULT,
    INSTALL_HEIGHT,
    REP_HOURS,
    SAMPLING_EPS,
    TRUNC_RAYS_DEFAULT,
    atmospheric_transmittance,
    cone_ray_directions,
    dni,
    precompute_position_terms,
    sample_mirror_points,
    sun_geometry,
)

MAP_GRID_STEP = 10.0  # 位置网格间距（m），η_trunc 空间变化平滑，10m 足够
CHUNK = 128           # 追迹分块大小（每块 chunk×S×K 光线，控制内存）


def trunc_free_batch(mirror_center, s, n_rays=TRUNC_RAYS_DEFAULT,
                     grid_n=GRID_N_DEFAULT, chunk=CHUNK):
    """无阴影假设的向量化截断效率（建表专用）。

    与 collector_trunc_efficiency 同口径的圆柱求交，但不传入 valid
    （建表假设无遮挡，分母为全部采样点×光线数；阴影损失由 eta_sb
    分项独立承担，二者乘法分解）。

    参数
    ----
    mirror_center : (M, 3) ndarray
        网格位置（镜心）坐标。
    s : (3,) ndarray
        太阳方向单位向量。
    n_rays, grid_n : int
        光锥光线数（2 的幂）、镜面采样网格数。
    chunk : int
        分块大小，控制 (chunk, S, K) 中间数组内存。

    返回
    ----
    (M,) ndarray：每位置的截断效率（无阴影口径）。
    """
    terms = precompute_position_terms(mirror_center)
    t = terms["t"]
    nv = s[None, :] + t
    n = nv / np.linalg.norm(nv, axis=1, keepdims=True)     # (M, 3) 法向

    dirs = cone_ray_directions(s, n_rays)                  # (K, 3)
    r2_cyl = COLLECTOR_RADIUS ** 2

    out = np.empty(len(mirror_center))
    for lo in range(0, len(mirror_center), chunk):
        mc = mirror_center[lo:lo + chunk]
        nc = n[lo:lo + chunk]
        P = sample_mirror_points(mc, nc, grid_n)           # (m, S, 3)

        cos_mk = nc @ dirs.T                               # (m, K)
        r = 2.0 * cos_mk[:, :, None] * nc[:, None, :] - dirs[None, :, :]  # (m, K, 3)

        A = r[:, :, 0] ** 2 + r[:, :, 1] ** 2              # (m, K)
        B = 2.0 * (P[:, :, None, 0] * r[:, None, :, 0]
                   + P[:, :, None, 1] * r[:, None, :, 1])  # (m, S, K)
        C = P[:, :, 0] ** 2 + P[:, :, 1] ** 2 - r2_cyl     # (m, S)
        disc = B ** 2 - 4.0 * A[:, None, :] * C[:, :, None]
        lam = (-B - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * A[:, None, :])
        z_hit = P[:, :, 2:3] + lam * r[:, None, :, 2]      # (m, S, K)

        hit = ((A > SAMPLING_EPS)[:, None, :] & (disc >= 0)
               & (lam > SAMPLING_EPS)
               & (z_hit >= COLLECTOR_Z_LOW) & (z_hit <= COLLECTOR_Z_HIGH))
        out[lo:lo + chunk] = hit.sum(axis=(1, 2)) / (P.shape[1] * n_rays)
    return out


def _attitude_cos_batch(mirror_center, s):
    """向量化计算各位置镜面的余弦效率 η_cos = s·n，n = (s+t)/|s+t|。"""
    terms = precompute_position_terms(mirror_center)
    nv = s[None, :] + terms["t"]
    n = nv / np.linalg.norm(nv, axis=1, keepdims=True)
    return np.clip(n @ s, 0.0, 1.0)


def _fill_nan_nearest(field):
    """将网格中的 nan（禁装区/场外）用最近有效值填充，保证插值平滑。"""
    from scipy.ndimage import distance_transform_edt

    valid = ~np.isnan(field)
    if valid.all():
        return field
    idx = distance_transform_edt(~valid, return_distances=False,
                                 return_indices=True)
    return field[tuple(idx)]


def build_efficiency_map(grid_step=MAP_GRID_STEP, n_rays=TRUNC_RAYS_DEFAULT,
                         grid_n=GRID_N_DEFAULT, save_path=None):
    """在镜场平面网格上预计算 60 时刻的效率地图并聚合年平均场。

    流程：
    1. 生成 [−350, 350]² 网格，保留 100 m ≤ r ≤ 350 m 的位置；
    2. 60 时刻循环：η_cos 解析计算，η_trunc 光锥追迹（无阴影口径）；
    3. 聚合：年平均分项、能量加权标量场 Φ；
    4. 禁装区/场外格点用最近有效值填充（避免插值 nan 传播）。

    返回
    ----
    dict：含全部 npz 字段（save_path 非空时另存 npz）。
    """
    xs = np.arange(-FIELD_RADIUS, FIELD_RADIUS + 1e-9, grid_step)
    ys = np.arange(-FIELD_RADIUS, FIELD_RADIUS + 1e-9, grid_step)
    X, Y = np.meshgrid(xs, ys)                             # (ny, nx)
    R = np.hypot(X, Y)
    mask = (R >= EXCLUSION_RADIUS) & (R <= FIELD_RADIUS)
    positions = np.column_stack([X[mask], Y[mask],
                                 np.full(mask.sum(), INSTALL_HEIGHT)])

    n_t = len(REP_HOURS) * 12                              # 60
    M = len(positions)
    eta_cos_t = np.empty((n_t, M))
    eta_trunc_t = np.empty((n_t, M))
    dni_t = np.empty(n_t)
    months, hours = [], []

    k = 0
    for month in range(1, 13):
        for hour in REP_HOURS:
            s = sun_geometry(month, hour)[4]
            eta_cos_t[k] = _attitude_cos_batch(positions, s)
            eta_trunc_t[k] = trunc_free_batch(positions, s, n_rays, grid_n)
            dni_t[k] = dni(s[2])
            months.append(month)
            hours.append(hour)
            k += 1

    d_hr = np.linalg.norm(COLLECTOR_CENTER[None, :] - positions, axis=1)
    eta_at_pos = atmospheric_transmittance(d_hr)

    # 聚合到位置维：能量加权标量与年平均分项
    phi_pos = np.mean(dni_t[:, None] * eta_cos_t * eta_trunc_t
                      * eta_at_pos[None, :], axis=0)
    cos_yr_pos = eta_cos_t.mean(axis=0)
    trunc_yr_pos = eta_trunc_t.mean(axis=0)

    def scatter(values):
        field = np.full(X.shape, np.nan)
        field[mask] = values
        return _fill_nan_nearest(field)

    data = {
        "xs": xs, "ys": ys,
        "mask": mask,
        "eta_cos_yr": scatter(cos_yr_pos),
        "eta_trunc_yr": scatter(trunc_yr_pos),
        "eta_at": scatter(eta_at_pos),
        "phi": scatter(phi_pos),
        "eta_cos_t": np.array([scatter(eta_cos_t[k]) for k in range(n_t)]),
        "eta_trunc_t": np.array([scatter(eta_trunc_t[k]) for k in range(n_t)]),
        "dni_t": dni_t,
        "months": np.array(months),
        "hours": np.array(hours),
        "params": np.array([grid_step, n_rays, grid_n]),
    }
    if save_path:
        np.savez_compressed(save_path, **data)
    return data


def load_efficiency_map(path):
    """加载 npz 效率地图。"""
    return dict(np.load(path, allow_pickle=False))


class EfficiencyMap:
    """效率地图查表接口（双线性插值）。

    用法
    ----
    em = EfficiencyMap("02_论文/结果/efficiency_map.npz")
    fields = em.query(mirror_xy)        # (N, 2) → dict of (N,)
    fields["phi"]                       # 能量加权年平均效率
    """

    def __init__(self, source):
        from scipy.interpolate import RegularGridInterpolator

        if isinstance(source, dict):
            data = source
        else:
            data = load_efficiency_map(source)
        self.xs, self.ys = data["xs"], data["ys"]

        def make(field):
            return RegularGridInterpolator(
                (self.ys, self.xs), field, method="linear",
                bounds_error=False, fill_value=None)  # 场外线性外推

        self._fields = {name: make(data[name]) for name in
                        ("phi", "eta_cos_yr", "eta_trunc_yr", "eta_at")}

    def query(self, mirror_xy):
        """对 (N, 2) 镜位置插值各效率场，返回 dict。"""
        pts = np.asarray(mirror_xy, dtype=float)
        return {name: f(pts) for name, f in self._fields.items()}
