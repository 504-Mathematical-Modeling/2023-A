"""问题一计算函数库：常量定义 + 各模块计算函数。

结构约定：
- 常量集中在本文件顶部，全大写命名。
- run_q1.py 只 import 函数，不重复定义常量。
- 函数按模块组织，职责单一、可独立测试。
"""

import numpy as np
import pandas as pd

# ============ 固定参数（问题一） ============
LAT_DEG = 39.4                 # 当地纬度（北纬，度）
ALT_KM = 3.0                   # 海拔（km）
SOLAR_CONST = 1.366            # 太阳常数 G0（kW/m2）

TOWER_XY = np.array([0.0, 0.0])            # 吸收塔位置（镜场坐标系，m）
COLLECTOR_CENTER = np.array([0.0, 0.0, 80.0])  # 集热器中心（m）
COLLECTOR_RADIUS = 3.5                      # 集热器圆柱半径（m）
COLLECTOR_Z_LOW = 76.0                      # 集热器受光高度下限（m）
COLLECTOR_Z_HIGH = 84.0                     # 集热器受光高度上限（m）

# 塔阴影参数（塔身与集热器合并为完整圆柱，假设半径与集热器相同）
TOWER_RADIUS = 3.5                          # 塔身半径（m）
TOWER_Z_LOW = 0.0                           # 塔底高度（m）
TOWER_Z_HIGH = 84.0                         # 塔顶高度（m）= 集热器顶部，含塔身[0,76]+集热器[76,84]

MIRROR_SIZE = 6.0             # 定日镜边长（m），正方形 6×6
MIRROR_AREA = MIRROR_SIZE**2  # 单镜面积（m2）
INSTALL_HEIGHT = 4.0          # 定日镜安装高度（m）
ETA_REF = 0.92                # 镜面反射率

FIELD_RADIUS = 350.0          # 镜场圆形区域半径（m）
EXCLUSION_RADIUS = 100.0      # 吸收塔周围禁装区半径（m）
MIN_SPACING = MIRROR_SIZE + 5.0  # 相邻底座最小间距（m），镜面宽度 + 5m

SUN_HALF_ANGLE_RAD = 4.65e-3  # 太阳锥形光束半角（rad）
NUM_DAYS = 12                 # 月数


def load_mirrors(path, mirror_xy_cols=(0, 1)):
    """读取附件中的定日镜坐标，构造镜面中心坐标。

    参数
    ----
    path : str
        附件 xlsx 路径。
    mirror_xy_cols : tuple
        坐标所在列号（附件首行为表头，自动跳过）。

    返回
    ----
    mirror_xy : (N, 2) ndarray
        镜面中心水平坐标（x, y），单位 m。
    mirror_center : (N, 3) ndarray
        镜面中心三维坐标（x, y, INSTALL_HEIGHT），单位 m。
    """
    raw = pd.read_excel(path, header=0)
    x = pd.to_numeric(raw.iloc[:, mirror_xy_cols[0]], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(raw.iloc[:, mirror_xy_cols[1]], errors="coerce").to_numpy(dtype=float)

    mirror_xy = np.column_stack([x, y])
    mirror_center = np.column_stack(
        [mirror_xy, np.full(len(mirror_xy), INSTALL_HEIGHT)]
    )
    return mirror_xy, mirror_center


def check_data_quality(mirror_xy):
    """检查附件坐标数据质量（缺失/重复/距离范围/相邻间距）。

    参数
    ----
    mirror_xy : (N, 2) ndarray
        镜面中心水平坐标。

    返回
    ----
    dict：各项检查结果。
    """
    from scipy.spatial import cKDTree

    report = {}
    report["total"] = len(mirror_xy)
    report["has_nan"] = bool(np.isnan(mirror_xy).any())

    _, counts = np.unique(mirror_xy, axis=0, return_counts=True)
    report["duplicates"] = int((counts > 1).sum())

    r = np.hypot(mirror_xy[:, 0], mirror_xy[:, 1])
    report["r_min"], report["r_max"] = float(r.min()), float(r.max())
    report["out_of_range"] = int(((r < EXCLUSION_RADIUS) | (r > FIELD_RADIUS)).sum())

    tree = cKDTree(mirror_xy)
    d, _ = tree.query(mirror_xy, k=2)
    report["min_spacing"] = float(d[:, 1].min())  # 最近邻距离最小值
    return report


def _set_cjk_font():
    """优先为 matplotlib 设置中文字体，避免中文乱码。"""
    import matplotlib
    from matplotlib import font_manager

    candidates = [
        "Noto Sans CJK SC", "Noto Serif CJK SC", "AR PL UMing CN",
        "SimHei", "Microsoft YaHei", "WenQuanYi Zen Hei",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            return
    matplotlib.rcParams["axes.unicode_minus"] = False


def plot_mirror_layout(mirror_xy, save_path=None):
    """绘制定日镜位置散点图，确认环形镜场布局。

    参数
    ----
    mirror_xy : (N, 2) ndarray
        镜面中心水平坐标。
    save_path : str | None
        非空时保存到该路径。
    """
    import matplotlib.pyplot as plt

    _set_cjk_font()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(mirror_xy[:, 0], mirror_xy[:, 1], s=1, c="tab:blue")
    ax.scatter(*TOWER_XY, c="tab:red", marker="*", s=120, label="吸收塔 (0,0)")
    circle_field = plt.Circle((0, 0), FIELD_RADIUS, fill=False, linestyle="--", color="k")
    circle_excl = plt.Circle((0, 0), EXCLUSION_RADIUS, fill=False, linestyle=":", color="gray")
    ax.add_patch(circle_field)
    ax.add_patch(circle_excl)
    ax.set_aspect("equal")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("定日镜场布局")
    ax.legend()
    ax.grid(alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_monthly_metrics(df1, save_path=None):
    """绘制月平均光学效率（含分项）与输出热功率随月份变化。

    上子图：总光学效率（黑实线）与余弦/阴影遮挡/截断三分量（灰阶虚线）；
    下子图：月平均输出热功率柱状图。两图共享月份横轴，避免 2×2 拼图
    与图内说明框造成的冗余构图。
    """
    import matplotlib.pyplot as plt

    _set_cjk_font()
    months = np.arange(1, 13)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.5, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0]})
    plt.subplots_adjust(hspace=0.14)

    for ax in (ax_top, ax_bot):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.7)
        ax.set_axisbelow(True)

    # 上：月平均光学效率及分项
    ax_top.plot(months, df1["eta"], color="#111111", lw=2.0,
                marker="o", ms=5, mfc="white", mec="#111111",
                label="总光学效率")
    ax_top.plot(months, df1["eta_cos"], color="#333333", lw=1.2,
                linestyle="-.", label="余弦效率")
    ax_top.plot(months, df1["eta_sb"], color="#666666", lw=1.2,
                linestyle="--", label="阴影遮挡效率")
    ax_top.plot(months, df1["eta_trunc"], color="#999999", lw=1.2,
                linestyle=":", label="截断效率")
    ax_top.set_ylabel("月平均光学效率")
    ax_top.set_ylim(0.40, 1.00)
    ax_top.legend(loc="lower center", bbox_to_anchor=(0.5, 1.03),
                  ncol=4, frameon=False, fontsize=9)

    # 下：月平均输出热功率柱状图
    ax_bot.bar(months, df1["p_mw"], width=0.62,
               color="#A6C4E8", edgecolor="#2F5597", linewidth=0.8)
    ax_bot.set_ylabel("月平均输出热功率 / MW")
    ax_bot.set_ylim(0, df1["p_mw"].max() * 1.22)
    ax_bot.set_xticks(months)
    ax_bot.set_xlabel("月份")
    for m, v in zip(months, df1["p_mw"]):
        ax_bot.text(m, v + 0.03, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color="#333333")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, (ax_top, ax_bot)


# ============ 太阳位置（每月 21 日） ============
# 每月 21 日相对春分（3 月 21 日，D=0）的天数 D
MONTH_21_D = [306, 337, 0, 31, 61, 92, 122, 153, 184, 214, 245, 275]
REP_HOURS = [9.0, 10.5, 12.0, 13.5, 15.0]  # 5 个代表时刻（当地时）


def day_from_spring_equinox(month):
    """返回月份 21 日相对春分（D=0）的天数 D。month: 1~12。"""
    return MONTH_21_D[month - 1]


def solar_declination(D):
    """太阳赤纬角 δ（rad），公式见题目附录：sin δ = sin(2πD/365)·sin(23.45°)。"""
    return np.arcsin(np.sin(2 * np.pi * D / 365) * np.sin(np.deg2rad(23.45)))


def hour_angle(hour):
    """太阳时角 ω（rad），ω = π/12·(ST − 12)。"""
    return np.pi / 12 * (hour - 12)


def sun_unit_vector(delta, omega, phi_deg=LAT_DEG):
    """太阳方向单位向量 s（x 东、y 北、z 上）与太阳高度角。

    直接构造向量，避免用 arccos 求方位角时丢失上午/下午信息。

    参数
    ----
    delta, omega : float
        赤纬角、时角，单位 rad。
    phi_deg : float
        当地纬度（度）。

    返回
    ----
    s : (3,) ndarray
        太阳方向单位向量，模长归一化为 1。
    alpha : float
        太阳高度角（rad）。
    """
    phi = np.deg2rad(phi_deg)
    sin_alpha = (
        np.sin(delta) * np.sin(phi)
        + np.cos(delta) * np.cos(phi) * np.cos(omega)
    )
    s = np.array([
        -np.cos(delta) * np.sin(omega),
        np.cos(phi) * np.sin(delta) - np.sin(phi) * np.cos(delta) * np.cos(omega),
        sin_alpha,
    ])
    return s / np.linalg.norm(s), np.arcsin(np.clip(sin_alpha, -1.0, 1.0))


def sun_geometry(month, hour):
    """某月 21 日某时刻的太阳几何参数。

    参数
    ----
    month : int
        1~12。
    hour : float
        当地时刻（h）。

    返回
    ----
    D : int
        相对春分天数。
    delta, omega : float
        赤纬角、时角（rad）。
    alpha_deg : float
        太阳高度角（度）。
    s : (3,) ndarray
        太阳方向单位向量。
    """
    D = day_from_spring_equinox(month)
    delta = solar_declination(D)
    omega = hour_angle(hour)
    s, alpha = sun_unit_vector(delta, omega)
    return D, delta, omega, np.degrees(alpha), s


# ============ DNI（法向直接辐射辐照度） ============
def dni_coeffs(alt_km=ALT_KM):
    """由海拔 H 计算 DNI 公式系数 a、b、c。

    a = 0.4237 − 0.00821(6−H)^2
    b = 0.5055 + 0.00595(6.5−H)^2
    c = 0.2711 + 0.01858(2.5−H)^2
    """
    a = 0.4237 - 0.00821 * (6 - alt_km) ** 2
    b = 0.5055 + 0.00595 * (6.5 - alt_km) ** 2
    c = 0.2711 + 0.01858 * (2.5 - alt_km) ** 2
    return a, b, c


def dni(sin_alpha, alt_km=ALT_KM):
    """法向直接辐射辐照度 DNI（kW/m2）。

    DNI = G0 * [a + b * exp(-c / sin(alpha_s))]，其中 G0 = SOLAR_CONST。

    参数
    ----
    sin_alpha : float 或 ndarray
        太阳高度角正弦，来自 sun_unit_vector/sun_geometry 的 s_z 分量。
    alt_km : float
        海拔高度（km）。

    返回
    ----
    float 或 ndarray
        DNI，单位 kW/m2。低太阳角时 exp(-c/sin_alpha) 趋 0，DNI 趋 G0*a。
    """
    a, b, c = dni_coeffs(alt_km)
    return SOLAR_CONST * (a + b * np.exp(-c / sin_alpha))


# ============ 镜面姿态与解析效率 ============
def atmospheric_transmittance(d_hr):
    """大气透射率 η_at，只依赖镜心到集热器中心的距离 d_HR（m）。

    η_at = 0.99321 − 0.0001176·d_HR + 1.97e-8·d_HR^2,  (d_HR ≤ 1000)
    """
    return 0.99321 - 0.0001176 * d_hr + 1.97e-8 * d_hr**2


def precompute_position_terms(mirror_center):
    """一次性计算并缓存仅依赖镜面位置的量，供 60 个时刻复用。

    参数
    ----
    mirror_center : (N, 3) ndarray
        镜面中心三维坐标。

    返回
    ----
    dict: {"t": (N,3), "d_hr": (N,), "eta_at": (N,)}
        t：镜心指向集热器中心的单位向量。
        d_hr：镜心到集热器中心的距离（m）。
        eta_at：大气透射率。
    """
    to_collector = COLLECTOR_CENTER - mirror_center
    d_hr = np.linalg.norm(to_collector, axis=1)
    t = to_collector / d_hr[:, None]
    eta_at = atmospheric_transmittance(d_hr)
    return {"t": t, "d_hr": d_hr, "eta_at": eta_at}


def mirror_attitude(mirror_center, s, position_terms=None):
    """由镜心坐标与太阳方向向量计算镜面朝向与解析效率（向量化）。

    参数
    ----
    mirror_center : (N, 3) ndarray
        镜面中心三维坐标（来自 load_mirrors）。
    s : (3,) ndarray
        太阳方向单位向量（来自 sun_unit_vector）。
    position_terms : dict | None
        precompute_position_terms 的结果；为 None 时内部自动计算一次。

    返回
    ----
    t : (N, 3) ndarray
        镜心指向集热器中心的单位向量。
    n : (N, 3) ndarray
        镜面法向单位向量（反射定律：平分 s 与 t）。
    eta_cos : (N,) ndarray
        余弦效率 η_cos = s·n。
    d_hr : (N,) ndarray
        镜心到集热器中心的距离（m）。
    eta_at : (N,) ndarray
        大气透射率。
    """
    if position_terms is None:
        position_terms = precompute_position_terms(mirror_center)
    t = position_terms["t"]
    d_hr = position_terms["d_hr"]
    eta_at = position_terms["eta_at"]

    sum_vec = s[None, :] + t
    n = sum_vec / np.linalg.norm(sum_vec, axis=1)[:, None]

    eta_cos = np.sum(s[None, :] * n, axis=1)
    return t, n, eta_cos, d_hr, eta_at


# ============ 阴影遮挡效率 ============
GRID_N_DEFAULT = 10          # 镜面采样网格数（每边），总采样点 GRID_N²
NEIGHBOR_RADIUS = 20.0       # 候选遮挡镜面筛选半径（m），= ⌈W√2 + d_max_NN⌉，见推导日志
SAMPLING_EPS = 1e-8          # 射线交点最小参数 λ（m），排除数值噪声与自身交点

def build_neighbors(mirror_center, radius=NEIGHBOR_RADIUS):
    """按镜心水平距离预计算每面镜的候选遮挡镜索引（不含自身）。

    阴影/挡光只可能由空间近邻镜面造成（低太阳角 14.4° 时阴影水平
    延伸约 12 m，见收敛测试），距离阈值筛选可大幅减少耗时较高的
    射线-矩形求交次数（距离矩阵构造本身仍为 O(N²)，但仅需一次
    预计算）。
    """
    d2 = np.sum((mirror_center[:, None, :2] - mirror_center[None, :, :2]) ** 2,
                axis=-1)
    mask = d2 <= radius ** 2
    np.fill_diagonal(mask, False)
    return [np.where(mask[i])[0] for i in range(len(mirror_center))]


def mirror_basis(n):
    """由镜面法向 n 构造镜面宽度方向 u 与高度方向 v（正交单位基）。

    题目约定镜面上下边始终平行于地面，故宽度方向 u 取法向水平投影
    旋转 90°：u = normalize(-ny, nx, 0)；高度方向 v = n × u。
    法向接近竖直（水平投影退化）时取 u = (1, 0, 0)。
    """
    norm_xy = np.hypot(n[:, 0], n[:, 1])
    u = np.zeros_like(n)
    ok = norm_xy > 1e-9
    u[ok, 0] = -n[ok, 1] / norm_xy[ok]
    u[ok, 1] = n[ok, 0] / norm_xy[ok]
    u[~ok, 0] = 1.0
    v = np.cross(n, u)
    return u, v


def sample_mirror_points(mirror_center, n, grid_n=GRID_N_DEFAULT):
    """生成每面镜面上的规则采样网格点。

    返回 (N, grid_n², 3)。格点取单元中心，避免边缘点落在镜面边界：
    frac = (i + 0.5)/grid_n − 0.5，i = 0..grid_n−1。
    """
    u, v = mirror_basis(n)
    frac = (np.arange(grid_n) + 0.5) / grid_n - 0.5
    du, dv = np.meshgrid(frac, frac)
    off = (du.ravel()[None, :, None] * MIRROR_SIZE * u[:, None, :]
           + dv.ravel()[None, :, None] * MIRROR_SIZE * v[:, None, :])
    return mirror_center[:, None, :] + off


def tower_shadow_mask(points, s, radius=TOWER_RADIUS,
                      z_lo=TOWER_Z_LOW, z_hi=TOWER_Z_HIGH):
    """判断采样点是否被塔（塔身+集热器合并圆柱 [z_lo, z_hi]）阴影遮挡。

    光线从采样点 P 沿 s 出发（指向太阳），与圆柱 x²+y²=R² 求交
    （水平分量二次方程），取最小正根 λ，检查交点高度是否落在圆柱
    高度范围内。若光线在到达太阳前先碰到塔体，则该点被塔阴影遮挡。

    参数
    ----
    points : (M, 3) ndarray
        镜面采样点坐标。
    s : (3,) ndarray
        太阳方向单位向量。

    返回
    ----
    (M,) bool ndarray：True 表示被塔阴影遮挡。
    """
    d = s
    A = d[0] ** 2 + d[1] ** 2
    hit = np.zeros(len(points), dtype=bool)
    if A <= 1e-12:  # 太阳接近天顶，光线无水平分量，塔阴影为零
        return hit
    B = 2.0 * (points[:, 0] * d[0] + points[:, 1] * d[1])
    C = points[:, 0] ** 2 + points[:, 1] ** 2 - radius ** 2
    disc = B ** 2 - 4.0 * A * C
    ok = disc >= 0
    lam = (-B - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * A)
    z_hit = points[:, 2] + lam * d[2]
    return ok & (lam > SAMPLING_EPS) & (z_hit >= z_lo) & (z_hit <= z_hi)


def _ray_rect_hit_batch(points, d, C, n, u, v, half=MIRROR_SIZE / 2):
    """批量判定采样点发出的光线是否被候选镜面集合中任一面遮挡。

    参数
    ----
    points : (S, 3) ndarray
        采样点坐标。
    d : (3,) ndarray
        光线方向单位向量。
    C, n, u, v : (K, 3) ndarray
        候选镜中心、法向、宽度基、高度基。

    返回
    ----
    (S,) bool ndarray：True 表示被任一候选镜面遮挡。
    """
    rel = C[None, :, :] - points[:, None, :]               # (S, K, 3)
    denom = d @ n.T                                        # (K,) 光线与镜面法向点积
    lam = np.sum(rel * n[None, :, :], axis=-1) / denom[None, :]  # (S, K)
    Q = points[:, None, :] + lam[..., None] * d            # (S, K, 3)
    in_rect = ((np.abs(np.sum((Q - C[None, :, :]) * u[None, :, :], axis=-1)) <= half)
               & (np.abs(np.sum((Q - C[None, :, :]) * v[None, :, :], axis=-1)) <= half))
    hit = (np.abs(denom) > 1e-9) & (lam > SAMPLING_EPS) & in_rect
    return hit.any(axis=1)


def effective_points(mirror_center, s, n, neighbors, grid_n=GRID_N_DEFAULT,
                     tower_radius=TOWER_RADIUS):
    """生成镜面采样点并计算未被阴影遮挡的有效掩码（张平论文三部分）。

    按优先级判定，避免重复扣除：
    1. 塔阴影（塔身+集热器圆柱）：优先级最高，被遮挡的点直接损失；
    2. 镜间阴影（沿 s 方向，指向太阳）：采样点沿 s 方向发射，若在
       到达太阳前先碰到其他镜面，则该点被前排镜面阴影遮挡；
    3. 镜间挡光（沿反射方向 r = 2(s·n)n − s）：反射光被前排镜面阻挡。

    参数
    ----
    mirror_center : (N, 3) ndarray
        镜面中心坐标。
    s : (3,) ndarray
        太阳方向单位向量。
    n : (N, 3) ndarray
        镜面法向单位向量。
    neighbors : list of ndarray
        build_neighbors 的预计算结果。
    grid_n : int
        镜面每边采样点数。
    tower_radius : float
        塔身圆柱半径（m），默认取全局常量 TOWER_RADIUS；
        供塔径敏感性分析传入不同取值。

    返回
    ----
    P : (N, grid_n², 3) ndarray
        每面镜的采样点坐标（截断效率复用，避免重复采样）。
    valid : (N, grid_n²) bool ndarray
        True 表示该采样点未被任何阴影遮挡（有效）。
    """
    P = sample_mirror_points(mirror_center, n, grid_n)   # (N, S, 3)
    S = P.shape[1]
    tower_hit = tower_shadow_mask(P.reshape(-1, 3), s,
                                  radius=tower_radius).reshape(-1, S)

    r = 2.0 * np.sum(s * n, axis=1)[:, None] * n - s     # 反射方向 (N, 3)
    u, v = mirror_basis(n)

    valid = ~tower_hit
    for i in range(len(mirror_center)):
        if neighbors[i].size:
            C_j = mirror_center[neighbors[i]]
            n_j = n[neighbors[i]]
            u_j = u[neighbors[i]]
            v_j = v[neighbors[i]]
            valid[i] &= ~_ray_rect_hit_batch(P[i], s, C_j, n_j, u_j, v_j)     # 阴影
            valid[i] &= ~_ray_rect_hit_batch(P[i], r[i], C_j, n_j, u_j, v_j)  # 挡光
    return P, valid


def shading_blocking_efficiency(mirror_center, s, n, neighbors,
                                grid_n=GRID_N_DEFAULT):
    """计算每面镜的阴影遮挡效率 eta_sb（张平论文三部分）。

    有效采样点比例 = 未被塔阴影、镜间阴影、镜间挡光遮挡的采样点比例。
    """
    _, valid = effective_points(mirror_center, s, n, neighbors, grid_n)
    return valid.mean(axis=1)


# ============ 截断效率（光锥光线追迹） ============
TRUNC_RAYS_DEFAULT = 1024     # 每采样点光锥内入射光线数（2 的幂，Sobol 平衡性最佳）
CONE_SIGMA_MAX = 4.65e-3     # 太阳锥形光束半角（rad），与 SUN_HALF_ANGLE_RAD 一致


def cone_ray_directions(s, n_rays=TRUNC_RAYS_DEFAULT):
    """光锥内入射方向（Sobol 准随机采样，张平论文式 9-11）。

    光锥坐标系：Z_S 沿主光线方向（+s，指向太阳圆盘中心），
    X_S 始终与地面平行（法向水平投影旋转 90°），Y_S = Z_S × X_S。
    任一光线在光锥坐标系：V_S = (sinσ·cosτ, sinσ·sinτ, cosσ)，
    其中 σ ∈ [0, 4.65 mrad] 为与主光线夹角，τ ∈ [0, 2π) 为周向角。

    σ = σ_max·√u₁（√ 保证圆盘面积均匀），τ = 2π·u₂，
    (u₁, u₂) 为 Sobol 二维准随机序列，结果可复现。

    参数
    ----
    s : (3,) ndarray
        太阳方向单位向量（主光线方向）。
    n_rays : int
        光线数，应为 2 的幂以保证 Sobol 平衡性。

    返回
    ----
    (n_rays, 3) ndarray：地面坐标系下的入射方向（从镜面指向太阳方向附近）。
    """
    from scipy.stats import qmc

    sampler = qmc.Sobol(d=2, scramble=False)
    u = sampler.random(n_rays)                    # (n_rays, 2)，u₁, u₂ ∈ (0,1)
    sigma = CONE_SIGMA_MAX * np.sqrt(u[:, 0])     # 面积均匀采样
    tau = 2.0 * np.pi * u[:, 1]

    # 光锥坐标系基
    zs = s / np.linalg.norm(s)                    # Z_S：主光线方向
    h = np.array([-s[1], s[0], 0.0])              # 水平投影旋转 90°
    h_norm = np.linalg.norm(h)
    xs = h / h_norm if h_norm > 1e-12 else np.array([1.0, 0.0, 0.0])
    ys = np.cross(zs, xs)                         # Y_S = Z_S × X_S

    # 光锥内光线方向（光锥坐标系）→ 地面坐标系
    vx = np.sin(sigma) * np.cos(tau)
    vy = np.sin(sigma) * np.sin(tau)
    vz = np.cos(sigma)
    return (vx[:, None] * xs + vy[:, None] * ys + vz[:, None] * zs)


def collector_trunc_efficiency(P, s, n, valid, n_rays=TRUNC_RAYS_DEFAULT):
    """计算每面镜的截断效率（Sobol 准随机光线追迹）。

    定义（题目附录）：ηtrunc = 集热器接收能量 / (镜面全反射能量 − 阴影遮挡损失能量)。
    分母 = 有效采样点发射的总光线数（已通过阴影遮挡判定），
    分子 = 其中反射光线到达集热器圆柱侧面的光线数。
    反射光线 r = 2(s'·n)·n − s'（s' 为光锥内该条入射光线方向），
    与圆柱 x²+y²=R² 求交（水平分量二次方程），检查交点高度 ∈ [76, 84] m。

    参数
    ----
    P : (N, S, 3) ndarray
        镜面采样点（来自 effective_points）。
    s : (3,) ndarray
        太阳方向单位向量（主光线方向）。
    n : (N, 3) ndarray
        镜面法向单位向量。
    valid : (N, S) bool ndarray
        有效采样点掩码（未被阴影遮挡）。
    n_rays : int
        每采样点光锥内光线数（2 的幂）。

    返回
    ----
    (N,) ndarray：每面镜的截断效率，范围 [0, 1]。
    """
    dirs = cone_ray_directions(s, n_rays)         # (K, 3) 入射方向
    eta_trunc = np.zeros(P.shape[0])

    for i in range(P.shape[0]):
        Pi = P[i]                                  # (S, 3)
        ni = n[i]
        valid_i = valid[i]
        if valid_i.sum() == 0:
            continue

        # 反射方向：r[k] = 2(s'_k·n_i)·n_i − s'_k，对每个采样点相同（平面镜）
        cos_theta = dirs @ ni                      # (K,)
        r = 2.0 * cos_theta[:, None] * ni - dirs   # (K, 3)

        # 圆柱求交（水平分量二次方程）
        A = r[:, 0] ** 2 + r[:, 1] ** 2            # (K,)
        A_ok = A > 1e-12                           # 光线平行于圆柱轴时无侧面交点
        B = 2.0 * (Pi[:, 0, None] * r[None, :, 0]
                   + Pi[:, 1, None] * r[None, :, 1])            # (S, K)
        C = Pi[:, 0] ** 2 + Pi[:, 1] ** 2 - COLLECTOR_RADIUS ** 2  # (S,)
        disc = B ** 2 - 4.0 * A[None, :] * C[:, None]            # (S, K)
        lam = (-B - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * A[None, :])  # (S, K)
        z_hit = Pi[:, 2, None] + lam * r[None, :, 2]             # (S, K)

        hit = (A_ok[None, :] & (disc >= 0) & (lam > SAMPLING_EPS)
               & (z_hit >= COLLECTOR_Z_LOW) & (z_hit <= COLLECTOR_Z_HIGH))
        hit[~valid_i] = False                      # 只统计有效采样点
        eta_trunc[i] = hit.sum() / (valid_i.sum() * n_rays)
    return eta_trunc


# ============ 汇总与平均 ============
def eta_field_weighted(eta_i, area):
    """镜场平均光学效率（按镜面面积加权）。问题一中各镜面积相等，等价于算术平均。"""
    return float(np.sum(area * eta_i) / np.sum(area))


def field_power_mw(dni_kw, area, eta_i):
    """镜场输出热功率 E_field（MW）。

    E_field = DNI * Σ A_i * η_i，DNI 单位 kW/m2，Σ 结果单位 kW，除以 1000 得 MW。
    """
    return float(dni_kw * np.sum(area * eta_i) / 1000.0)


def mirror_optical_efficiency(eta_cos, eta_at, eta_sb=1.0, eta_trunc=1.0,
                              eta_ref=ETA_REF):
    """单镜光学效率 η_i = η_sb * η_cos * η_at * η_trunc * η_ref。"""
    return eta_sb * eta_cos * eta_at * eta_trunc * eta_ref