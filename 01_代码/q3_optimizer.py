"""问题三优化求解器：异尺寸定日镜场（环形带分区 + 带约束贪心 + 差分进化）。

优化模型（与论文 5.3 一致）：
    max  q = P̄ / A_total        （等价 min A_total）
    s.t. P̄ ≥ 60 MW（60 个代表时刻等权平均口径）
         |p_i| ≤ 350 m，|p_i − T| ≥ 100 m
         |p_i − p_j| ≥ max(w_i, w_j) + 5 m
         2 ≤ w_i, H_i ≤ 8 m，H_i ≤ w_i，2 ≤ a_i ≤ 6 m，a_i ≥ H_i/2

求解框架：
    外层：差分进化（DE）在 (T, w_k, H_k, a_k)_{k=1..3} 11 维空间搜索；
          环形带 K=3 按候选镜到塔距离分位划分，带内尺寸统一；
    内层：每带独立六角晶格候选（间距 d_k = w_k + 5）→ 全局解析贡献贪心选镜
          （带约束：候选点冲突图跳选，保证跨带间距约束）；
    两级精度：DE 内低精度 6×6/32；精英 8×8/512 复核；最终 10×10/1024 硬检查。
"""

import os
import sys

import functools
import numpy as np
import pandas as pd

from common import (
    A_MAX,
    A_MIN,
    ETA_REF,
    FIELD_RADIUS,
    FINAL_GRID,
    FINAL_RAYS,
    GREEDY_DISCOUNT,
    LAMBDA,
    LATTICE_MARGIN,
    OUTPUT_DIR,
    RATED_POWER_MW,
    W_MAX,
    W_MIN,
    analytic_contribution,
    field_eval,
    hexagonal_lattice,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(ROOT, "02_论文", "figures")
Q2_RESULT = os.path.join(OUTPUT_DIR, "q2_result.npy")

K_BANDS = 3                     # 环形带数
DE_POP3, DE_MAXITER3, DE_SEED3 = 10, 40, 2023
DE_TARGET_MW = RATED_POWER_MW + 2.0   # DE 低精度目标（62 MW，抵消低精度高估 ~0.6% + 复核裕量）
REVIEW_GRID, REVIEW_RAYS = 8, 512
GREEDY_SCALE = 1.02             # 贪心目标补偿（低精度高估 ~0.6% + 复核裕量）
MAX_ADJUST = 8                  # 复核补镜/删镜最大轮数


@functools.lru_cache(maxsize=1)
def load_q2_params():
    """读取问题二解的晶格参数 (θ, phase_x, phase_y)；不存在时用默认值。"""
    if os.path.exists(Q2_RESULT):
        try:
            x = np.load(Q2_RESULT, allow_pickle=True).item()["x"]
            return float(x[5]), float(x[6]), float(x[7])
        except Exception:
            pass
    return 0.52, 0.0, 0.0


def band_assign(cand_xy, tower, n_bands=K_BANDS):
    """按 |p − T| 的等点数分位把候选点分配到环形带，返回带号数组。"""
    r = np.hypot(cand_xy[:, 0] - tower[0], cand_xy[:, 1] - tower[1])
    order = np.argsort(r)
    band = np.empty(len(cand_xy), dtype=int)
    n = len(cand_xy)
    for k in range(n_bands):
        lo = (order.size * k) // n_bands
        hi = (order.size * (k + 1)) // n_bands
        band[order[lo:hi]] = k
    return band


def clip_x(x):
    """把 11 维决策向量裁剪到可行域：2≤H≤w≤8、2≤a≤6、a≥H/2。"""
    tx, ty = x[0], x[1]
    bands = [(x[2], x[3], x[4]), (x[5], x[6], x[7]), (x[8], x[9], x[10])]
    clipped = []
    for wk, Hk, ak in bands:
        wk = float(np.clip(wk, W_MIN, W_MAX))
        Hk = float(np.clip(min(Hk, wk), W_MIN, W_MAX))
        ak = float(np.clip(ak, A_MIN, A_MAX))
        ak = max(ak, Hk / 2.0)
        clipped += [wk, Hk, ak]
    return np.array([tx, ty] + clipped)


def build_layout(x, theta, phase):
    """由 11 维决策变量构建候选布局。

    返回 (cand_xy, band, w_p, H_p, a_p, c)：
    cand_xy  候选镜位（各带晶格合并）；
    band     每候选点所属带号；
    w_p/H_p/a_p  每候选点的尺寸与安装高度（按带）；
    c        解析年均单位面积贡献（kW/m²，用于贪心排序）。
    """
    tx, ty = x[0], x[1]
    bands = [(x[2], x[3], x[4]), (x[5], x[6], x[7]), (x[8], x[9], x[10])]
    tower = np.array([tx, ty])

    cand_list, w_list, H_list, a_list = [], [], [], []
    for k, (wk, Hk, ak) in enumerate(bands):
        dk = wk + LATTICE_MARGIN
        pk = hexagonal_lattice(tower, dk, theta, (phase[0] % dk, phase[1] % dk))
        cand_list.append(pk)
        w_list.append(np.full(len(pk), wk))
        H_list.append(np.full(len(pk), Hk))
        a_list.append(np.full(len(pk), ak))

    cand_xy = np.vstack(cand_list)
    band = band_assign(cand_xy, tower)
    w_p = np.concatenate(w_list)
    H_p = np.concatenate(H_list)
    a_p = np.concatenate(a_list)
    c = analytic_contribution(cand_xy, tower, a_p)
    return cand_xy, band, w_p, H_p, a_p, c


def conflict_table(cand_xy, w_p, H_p, max_radius=13.0):
    """候选点冲突邻接表：两候选间距 < max(w_i, w_j) + 5 时互斥。

    带内已由晶格间距 d_k = w_k + 5 保证无冲突，跨带交叉冲突在此建表。
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(cand_xy)
    pairs = tree.query_pairs(max_radius)
    adj = [[] for _ in range(len(cand_xy))]
    for i, j in pairs:
        lim = max(w_p[i], w_p[j]) + LATTICE_MARGIN
        if np.hypot(cand_xy[i, 0] - cand_xy[j, 0],
                    cand_xy[i, 1] - cand_xy[j, 1]) < lim - 1e-6:
            adj[i].append(j)
            adj[j].append(i)
    return adj


def greedy3(cand_xy, band, w_p, H_p, c, tower):
    """带约束贪心选镜：按解析贡献 c 降序，跳过与已选镜冲突的候选。

    累计目标 = 额定功率 / (η_ref × 折扣) × 补偿系数（kW）。
    返回 (选中索引按 c 降序, 目标功率 kW)。
    """
    target_kw = (RATED_POWER_MW * 1000.0 * GREEDY_SCALE
                 / (ETA_REF * GREEDY_DISCOUNT))
    area_p = w_p * H_p
    order = np.argsort(-c)
    adj = conflict_table(cand_xy, w_p, H_p)
    blocked = np.zeros(len(cand_xy), dtype=bool)
    picked = []
    acc = 0.0
    for i in order:
        if blocked[i]:
            continue
        acc += c[i] * area_p[i]
        picked.append(i)
        if acc >= target_kw:
            break
        for j in adj[i]:
            blocked[j] = True
    return picked, target_kw


def fitness3(x, grid_n=5, n_rays=16):
    """DE 适应度：低精度 60 时刻追迹功率 + 总面积，罚函数处理功率缺口。

    用 5×5/16 快速口径（与 6×6/32 排序一致，略高估 +0.2~0.3%），
    DE_TARGET_MW=62 已含补偿裕量，最终由 8×8/512 复核与 10×10/1024 验收。
    """
    tx, ty = x[0], x[1]
    xv = clip_x(x)
    theta, phx, phy = load_q2_params()
    cand_xy, band, w_p, H_p, a_p, c = build_layout(xv, theta, (phx, phy))
    if len(cand_xy) < 200:
        return 1e12

    picked, _ = greedy3(cand_xy, band, w_p, H_p, c, np.array([tx, ty]))
    if len(picked) == 0:
        return 1e12
    sel = cand_xy[picked]
    P_est, _ = field_eval(sel, np.array([tx, ty]), w_p[picked], H_p[picked],
                          a_p[picked], grid_n, n_rays)
    area = float(np.sum(w_p[picked] * H_p[picked]))
    gap = max(0.0, DE_TARGET_MW - P_est)
    return area + LAMBDA * gap ** 2


def run_de3(workers=1, log_path=None, x0=None):
    """差分进化搜索 11 维决策空间（塔位 + 3 带 × (w, H, a)）。

    若 log_path 已有检查点，则把历史最优个体作为 x0 注入初始种群（热启动），
    便于系统重启后续跑（检查点每次 callback 落盘）。
    """
    from scipy.optimize import differential_evolution

    bounds = [(-320.0, 320.0), (-320.0, 320.0)]
    for _ in range(K_BANDS):
        bounds += [(W_MIN, W_MAX), (W_MIN, W_MAX), (A_MIN, A_MAX)]

    history, best_per_gen = [], []
    if x0 is None:
        x0 = None
        if log_path and os.path.exists(log_path):
            try:
                hist = np.load(log_path, allow_pickle=True)
                if len(hist) > 0:
                    Fs = np.array([h[1] for h in hist])
                    x0 = hist[int(np.argmin(Fs))][0]
                    print(f"  检查点恢复: {len(hist)} 代历史, 历史最优 F = {Fs.min():.1f}, "
                          f"将其作为 x0 注入初始种群", flush=True)
            except Exception as e:
                print(f"  检查点读取失败({e})，全新启动", flush=True)
    else:
        print(f"  注入自定义 x0（问题二解派生）: 塔({x0[0]:.1f},{x0[1]:.1f}) "
              f"带1 {x0[2]:.2f}×{x0[3]:.2f}", flush=True)

    def callback(xk, convergence):
        F = fitness3(xk)
        history.append((xk.copy(), F, convergence))
        best_per_gen.append(xk.copy())
        if log_path:
            np.save(log_path, np.array(history, dtype=object))
        print(f"    [DE 进度] 第 {len(history)} 代结束, "
              f"当前最优 F = {F:.1f}", flush=True)

    print(f"  DE: popsize={DE_POP3}, maxiter={DE_MAXITER3}, "
          f"seed={DE_SEED3}, workers={workers}, dim={len(bounds)}", flush=True)
    kwargs = dict(
        strategy="best1bin", maxiter=DE_MAXITER3, popsize=DE_POP3,
        seed=DE_SEED3, tol=0.0, workers=workers,
        callback=callback, polish=False)
    if x0 is not None:
        kwargs["x0"] = x0
    result = differential_evolution(fitness3, bounds=bounds, **kwargs)
    return result, best_per_gen


def final_review3(x, grid_n=REVIEW_GRID, n_rays=REVIEW_RAYS,
                  max_adjust=MAX_ADJUST):
    """精英解复核与局部调整（8×8/512）：功率不足补镜、超标删镜。"""
    tx, ty = x[0], x[1]
    bands = [(x[2], x[3], x[4]), (x[5], x[6], x[7]), (x[8], x[9], x[10])]
    clipped = []
    for wk, Hk, ak in bands:
        wk = float(np.clip(wk, W_MIN, W_MAX))
        Hk = float(np.clip(min(Hk, wk), W_MIN, W_MAX))
        ak = float(np.clip(ak, A_MIN, A_MAX))
        ak = max(ak, Hk / 2.0)
        clipped += [wk, Hk, ak]
    xv = np.array([tx, ty] + clipped)
    tower = np.array([tx, ty])
    theta, phx, phy = load_q2_params()
    cand_xy, band, w_p, H_p, a_p, c = build_layout(xv, theta, (phx, phy))

    picked, _ = greedy3(cand_xy, band, w_p, H_p, c, tower)
    picked = list(picked)
    order = np.argsort(-c)
    tail = [i for i in order if i not in set(picked)]

    adj = conflict_table(cand_xy, w_p, H_p)
    for _ in range(max_adjust):
        P_est, eff = field_eval(cand_xy[picked], tower, w_p[picked],
                                H_p[picked], a_p[picked], grid_n, n_rays)
        if P_est < RATED_POWER_MW and tail:
            nxt = tail.pop(0)
            if any(j in picked for j in adj[nxt]):
                continue
            picked.append(nxt)
        elif P_est > RATED_POWER_MW + 1.0 and len(picked) > 100:
            picked.pop()  # 按 c 降序最后选入者贡献最低
        else:
            break
    P_final, eff = field_eval(cand_xy[picked], tower, w_p[picked],
                              H_p[picked], a_p[picked], grid_n, n_rays)
    return cand_xy[picked], tower, w_p[picked], H_p[picked], a_p[picked], \
        P_final, eff


def month_eval(sel_xy, tower, w_arr, H_arr, a_arr, month, grid_n, n_rays):
    """单月 5 个代表时刻追迹，返回该月平均 (P_MW, 分项效率 dict)。"""
    import numpy as np

    from q1_functions import REP_HOURS, dni, sun_geometry
    from common import (
        ETA_REF,
        neighbor_radius,
        param_attitude,
        param_effective_points,
        param_neighbors,
        param_terms,
        param_trunc,
    )

    a_arr = np.broadcast_to(a_arr, (len(sel_xy),))
    w_arr = np.broadcast_to(w_arr, (len(sel_xy),))
    H_arr = np.broadcast_to(H_arr, (len(sel_xy),))
    mc = np.column_stack([sel_xy, a_arr])
    area = w_arr * H_arr

    radius = neighbor_radius(float(np.max(w_arr)))
    neighbors = param_neighbors(sel_xy, radius)
    terms = param_terms(mc, tower)
    acc_p = acc_cos = acc_sb = acc_trunc = acc_eta = 0.0
    for hour in REP_HOURS:
        s = sun_geometry(month, hour)[4]
        _, n, eta_cos, _, eta_at = param_attitude(mc, s, tower, terms)
        P, valid = param_effective_points(mc, s, n, neighbors, w_arr, H_arr,
                                          tower, grid_n)
        eta_sb = valid.mean(axis=1)
        eta_trunc = param_trunc(P, s, n, valid, tower, n_rays)
        eta_i = ETA_REF * eta_cos * eta_at * eta_sb * eta_trunc
        acc_p += dni(s[2]) * np.sum(area * eta_i) / 1000.0
        acc_cos += np.mean(eta_cos)
        acc_sb += np.mean(eta_sb)
        acc_trunc += np.mean(eta_trunc)
        acc_eta += np.sum(eta_i) / len(eta_i)
    n_t = len(REP_HOURS)
    return (acc_p / n_t,
            {"eta_cos": acc_cos / n_t, "eta_sb": acc_sb / n_t,
             "eta_trunc": acc_trunc / n_t, "eta": acc_eta / n_t})


def export_result3(sel_xy, tower, w_arr, H_arr, a_arr, path=None):
    """按模板导出 result3.xlsx（每镜宽/高/安装高逐镜填写）。"""
    if path is None:
        path = os.path.join(ROOT, "00_题目与数据", "result3.xlsx")
    df = pd.DataFrame({
        "吸收塔x坐标 (m)": float(tower[0]),
        "吸收塔y坐标 (m)": float(tower[1]),
        "定日镜序号": np.arange(1, len(sel_xy) + 1),
        "定日镜宽度 (m)": np.asarray(w_arr),
        "定日镜高度 (m)": np.asarray(H_arr),
        "定日镜x坐标 (m)": sel_xy[:, 0],
        "定日镜y坐标 (m)": sel_xy[:, 1],
        "定日镜z坐标 (m)": np.asarray(a_arr),
    })
    df.to_excel(path, index=False)
    return path


def export_tables(sel_xy, tower, w_arr, H_arr, a_arr, P_mw, eff):
    """导出表 1（逐月）、表 2（年均）、表 3（设计参数）csv 到 02_论文/files/。"""
    A_total = float(np.sum(w_arr * H_arr))
    rows = []
    for month in range(1, 13):
        P_m, em = month_eval(sel_xy, tower, w_arr, H_arr, a_arr, month,
                             FINAL_GRID, FINAL_RAYS)
        rows.append((month, em["eta_cos"], em["eta_sb"], em["eta_trunc"],
                     em["eta"], P_m, P_m * 1000.0 / A_total))
        print(f"  表 1: {month} 月 P={P_m:6.2f}MW", flush=True)
    df1 = pd.DataFrame(rows, columns=["日期", "eta_cos", "eta_sb",
                                      "eta_trunc", "eta", "p_mw", "q"])
    df1["日期"] = df1["日期"].apply(lambda m: f"{m} 月 21 日")
    df1.to_csv(os.path.join(OUTPUT_DIR, "q3_表1_每月平均.csv"), index=False)

    q = P_mw * 1000.0 / A_total
    df2 = pd.DataFrame([{**{k: v for k, v in eff.items()},
                         "p_mw": P_mw, "q": q}])
    df2.to_csv(os.path.join(OUTPUT_DIR, "q3_表2_年平均.csv"), index=False)

    df3 = pd.DataFrame([{
        "吸收塔位置坐标": f"({tower[0]:.1f}, {tower[1]:.1f})",
        "定日镜尺寸（宽×高）": "各异（见 result3.xlsx）",
        "定日镜安装高度(m)": "各异（见 result3.xlsx）",
        "定日镜总面数": len(sel_xy),
        "定日镜总面积(m2)": round(A_total, 1),
    }])
    df3.to_csv(os.path.join(OUTPUT_DIR, "q3_表3_设计参数.csv"), index=False)
    return df1, df2, df3


def plot_q3_layout(sel_xy, tower, band_id, save_path=None):
    """镜场布局俯视图（按环形带着色）。"""
    import matplotlib.pyplot as plt

    from q1_functions import _set_cjk_font
    from common import EXCLUSION_RADIUS, FIELD_RADIUS

    _set_cjk_font()
    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = {0: "tab:blue", 1: "tab:green", 2: "tab:orange"}
    for k in range(K_BANDS):
        m = band_id == k
        ax.scatter(sel_xy[m, 0], sel_xy[m, 1], s=0.8, c=cmap[k],
                   label=f"第 {k + 1} 环带（{m.sum()} 面）")
    ax.scatter(*tower, c="tab:red", marker="*", s=160, label="吸收塔")
    ax.add_patch(plt.Circle((0, 0), FIELD_RADIUS, fill=False, ls="--", color="k"))
    ax.add_patch(plt.Circle(tower, EXCLUSION_RADIUS, fill=False, ls=":",
                            color="gray"))
    ax.set_aspect("equal")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title(f"问题三优化布局（N={len(sel_xy)}，塔位 ({tower[0]:.1f}, {tower[1]:.1f})）")
    ax.legend(loc="upper right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def export_all(summary, sel):
    """从已算好的 summary 与镜位重建全部导出产物。"""
    tower = summary["tower"]
    w_arr = summary["w"]
    H_arr = summary["H"]
    a_arr = summary["a"]
    band_id = summary["band"]
    P_final = summary["P_mw"]
    eff = {k: summary[k] for k in ("eta_cos", "eta_sb", "eta_trunc", "eta")}
    A_total = summary["A_total"]
    q = summary["q"]
    path = export_result3(sel, tower, w_arr, H_arr, a_arr)
    print(f"  result3.xlsx → {path}")
    export_tables(sel, tower, w_arr, H_arr, a_arr, P_final, eff)
    fig = plot_q3_layout(sel, tower, band_id,
                         os.path.join(FIGURE_DIR, "fig_q3_layout.png"))
    print(f"  布局图 → {fig}")
    print("  结果已缓存 → 02_论文/files/q3_result.npy, q3_sel.npy")
    return summary


def main():
    print("=" * 60)
    print("问题三：异尺寸定日镜场优化设计（环形带分区 + 贪心 + 差分进化）")
    print("=" * 60)

    force_rerun = "--rerun" in sys.argv
    if force_rerun:
        print("  --rerun 指定：跳过缓存检查，重新求解", flush=True)

    theta, phx, phy = load_q2_params()
    print(f"  晶格参数（取自问题二解）: θ={theta:.4f}, phase=({phx:.2f}, {phy:.2f})")

    RESULT_PATH = os.path.join(OUTPUT_DIR, "q3_result.npy")
    SEL_PATH = os.path.join(OUTPUT_DIR, "q3_sel.npy")
    if not force_rerun and os.path.exists(RESULT_PATH) and os.path.exists(SEL_PATH):
        try:
            summary = np.load(RESULT_PATH, allow_pickle=True).item()
            sel = np.load(SEL_PATH)
            if summary["P_mw"] >= RATED_POWER_MW:
                print("发现已达标缓存结果，跳过 DE 与复核，直接导出", flush=True)
                return export_all(summary, sel)
        except Exception as e:
            print(f"  缓存读取失败({e})，重新求解", flush=True)

    x0 = None
    q2_path = os.path.join(OUTPUT_DIR, "q2_result.npy")
    if os.path.exists(q2_path):
        try:
            qx = np.load(q2_path, allow_pickle=True).item()["x"]
            a0 = max(float(qx[4]), float(qx[3]) / 2.0)
            x0 = np.array([qx[0], qx[1]] + list(qx[2:5]) * K_BANDS)
            x0[4], x0[7], x0[10] = a0, a0, a0
            if os.path.exists(os.path.join(ROOT, "01_代码", "q3_de_history.npy")):
                os.remove(os.path.join(ROOT, "01_代码", "q3_de_history.npy"))
                print("  已清除旧 DE 检查点（改用问题二解派生 x0）", flush=True)
        except Exception as e:
            print(f"  问题二解读取失败({e})，无自定义 x0", flush=True)

    result, best_per_gen = run_de3(workers=16, x0=x0,
                                   log_path=os.path.join(ROOT, "01_代码", "q3_de_history.npy"))
    print(f"\nDE 完成: 迭代 {DE_MAXITER3} 代, 最优适应度 F = {result.fun:.1f}")

    # 精英筛选：每代最优 + DE 最终解，去重后低精度排序
    candidates = list(best_per_gen) + [result.x]
    scored = []
    seen = set()
    for xk in candidates:
        key = tuple(np.round(xk, 3))
        if key in seen:
            continue
        seen.add(key)
        xv = clip_x(xk)
        tx, ty = xv[0], xv[1]
        cand_xy, band, w_p, H_p, a_p, c = build_layout(
            xv, theta, (phx, phy))
        picked, _ = greedy3(cand_xy, band, w_p, H_p, c, np.array([tx, ty]))
        P_est, _ = field_eval(cand_xy[picked], np.array([tx, ty]),
                              w_p[picked], H_p[picked], a_p[picked], 6, 32)
        A = float(np.sum(w_p[picked] * H_p[picked]))
        scored.append((xk, P_est, A))
        print(f"  候选: 塔({tx:7.1f},{ty:7.1f}) 带1 {xv[2]:.2f}×{xv[3]:.2f} "
              f"带2 {xv[5]:.2f}×{xv[6]:.2f} 带3 {xv[8]:.2f}×{xv[9]:.2f} "
              f"N={len(picked):5d} A={A:9.1f} P={P_est:6.2f}MW")

    scored.sort(key=lambda t: (t[1] < RATED_POWER_MW, t[2]))

    chosen = None
    for xk, P_low, A_low in scored[:3]:
        print(f"\n复核（60 时刻，8×8 网格，512 光线）: "
              f"塔({xk[0]:.1f},{xk[1]:.1f}) ...", flush=True)
        sel, tower, w_arr, H_arr, a_arr, P_final, eff = final_review3(xk)
        if P_final >= RATED_POWER_MW:
            chosen = (sel, tower, w_arr, H_arr, a_arr, P_final, eff, xk)
            break
        print(f"  → 复核 P={P_final:.2f}MW 未达标，尝试下一候选", flush=True)

    if chosen is None:
        raise RuntimeError("所有精英复核均未达到 60MW，需增加 DE 迭代或放宽约束")

    sel, tower, w_arr, H_arr, a_arr, P_final, eff, best_x = chosen

    print("\n最终高精度评价（10×10 网格，1024 光线）...", flush=True)
    P_final, eff = field_eval(sel, tower, w_arr, H_arr, a_arr,
                              FINAL_GRID, FINAL_RAYS)
    if P_final < RATED_POWER_MW:
        raise RuntimeError(f"最终高精度功率 {P_final:.2f}MW < 60MW，方案不合格")

    A_total = float(np.sum(w_arr * H_arr))
    q = P_final * 1000.0 / A_total
    r = np.hypot(sel[:, 0] - tower[0], sel[:, 1] - tower[1])
    band_id = band_assign(sel, tower)

    print("\n======== 问题三最终结果 ========")
    print(f"  吸收塔位置: ({tower[0]:.1f}, {tower[1]:.1f}) m")
    print(f"  定日镜数: {len(sel)}, 镜面总面积: {A_total:.1f} m²")
    print(f"  年平均输出热功率: {P_final:.3f} MW")
    print(f"  单位镜面面积年平均输出热功率: {q:.4f} kW/m²")
    print(f"  年均分项效率: 余弦 {eff['eta_cos']:.4f}, "
          f"阴影遮挡 {eff['eta_sb']:.4f}, 截断 {eff['eta_trunc']:.4f}, "
          f"综合 {eff['eta']:.4f}")
    for k in range(K_BANDS):
        m = band_id == k
        print(f"  带 {k + 1}: {m.sum():4d} 面, w∈[{w_arr[m].min():.2f},{w_arr[m].max():.2f}] "
              f"H∈[{H_arr[m].min():.2f},{H_arr[m].max():.2f}] "
              f"a∈[{a_arr[m].min():.2f},{a_arr[m].max():.2f}] "
              f"r∈[{r[m].min():.1f},{r[m].max():.1f}] m")

    summary = {
        "tower": tower, "w": w_arr, "H": H_arr, "a": a_arr,
        "band": band_id, "N": len(sel), "A_total": A_total,
        "P_mw": P_final, "q": q, **eff, "x": np.asarray(best_x),
    }
    np.save(os.path.join(OUTPUT_DIR, "q3_result.npy"), summary)
    np.save(os.path.join(OUTPUT_DIR, "q3_sel.npy"), sel)
    return export_all(summary, sel)
    return summary


if __name__ == "__main__":
    main()
