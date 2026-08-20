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

追迹评价、布局生成等通用组件见 common.py。
"""

import os

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from common import (
    A_MAX,
    A_MIN,
    ETA_REF,
    EXCLUSION_RADIUS,
    FAST_GRID,
    FAST_RAYS,
    FIELD_RADIUS,
    FIGURE_DIR,
    FINAL_GRID,
    FINAL_RAYS,
    GREEDY_DISCOUNT,
    LAMBDA,
    LATTICE_MARGIN,
    OUTPUT_DIR,
    P_FINAL_TOL,
    RATED_POWER_MW,
    ROOT,
    W_MAX,
    W_MIN,
    analytic_contribution,
    field_eval,
    greedy_select,
    hexagonal_lattice,
)

# ============ 问题二专属参数 ============
DE_POP, DE_MAXITER, DE_SEED = 8, 20, 2023
DE_TARGET_MW = RATED_POWER_MW + 1.2   # DE 低精度目标（抵消低精度高估 ~0.6%）


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
    gap = max(0.0, DE_TARGET_MW - P_est)
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
        if log_path:
            np.save(log_path, np.array(history, dtype=object))
        print(f"    [DE 进度] 第 {len(history)} 代结束, "
              f"当前最优 F = {F:.1f}", flush=True)

    print(f"  DE: popsize={popsize}, maxiter={maxiter}, "
          f"seed={seed}, workers={workers}", flush=True)
    result = differential_evolution(
        fitness, bounds=bounds, strategy="best1bin",
        maxiter=maxiter, popsize=popsize, seed=seed,
        tol=0.0, workers=workers, callback=callback,
        polish=False)
    print(f"  DE 完成，共评价 {len(history)} 个个体", flush=True)
    return result, best_per_gen


def final_review(best_x, grid_n=8, n_rays=512,
                 max_adjust=8):
    """最优解的复核与局部调整（60 时刻追迹，默认 8×8/512 中等精度）。

    功率不足 → 按解析贡献补镜；超标过多 → 删最低贡献镜。
    复核调整循环内使用中等精度以保证速度，最终报告精度由调用方
    另行以 10×10/1024 高精度复核一次。
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

    result, best_per_gen = run_de(workers=24,
                                  log_path="/tmp/opencode/q2_de_history.npy")
    print(f"\nDE 完成: 迭代 {DE_MAXITER} 代, 最优适应度 F = {result.fun:.1f}")
    print(f"  DE 最优: T = ({result.x[0]:.2f}, {result.x[1]:.2f}) m, "
          f"镜面 {result.x[2]:.2f}×{result.x[3]:.2f} m, "
          f"安装高 {result.x[4]:.2f} m, θ={result.x[5]:.3f}")

    # 精英筛选：每代最优个体 + DE 最终解，用低精度 60 时刻精确追迹排序
    candidates = list(best_per_gen)
    candidates.append(result.x)
    scored = []
    seen = set()
    for xk in candidates:
        key = tuple(np.round(xk, 3))
        if key in seen:
            continue
        seen.add(key)
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

    # 高精度复核（按低精度排序取前 3，8×8/512 中等精度调整）
    chosen = None
    for xk, sel, tower, w, H, a, P_low, A_low in scored[:3]:
        print(f"\n复核（60 时刻，8×8 网格，512 光线）: "
              f"塔({tower[0]:.1f},{tower[1]:.1f}) ...", flush=True)
        sel_f, tower_f, w_f, H_f, a_f, P_final, eff = final_review(
            xk)
        if P_final >= RATED_POWER_MW - P_FINAL_TOL:
            chosen = (sel_f, tower_f, w_f, H_f, a_f, P_final, eff, xk)
            break
        print(f"  → 复核 P={P_final:.2f}MW 未达标，尝试下一候选", flush=True)

    if chosen is None:
        raise RuntimeError("所有精英复核均未达到 60MW，需增加 DE 迭代或放宽约束")

    sel, tower, w, H, a, P_final, eff, best_x = chosen

    # 最终方案做一次 10×10/1024 高精度报告评价
    print("\n最终高精度评价（10×10 网格，1024 光线）...", flush=True)
    P_final, eff = field_eval(sel, tower, w, H, a,
                              FINAL_GRID, FINAL_RAYS)
    if P_final < RATED_POWER_MW:
        raise RuntimeError(
            f"最终高精度评价 P̄={P_final:.3f} MW < {RATED_POWER_MW} MW "
            "未达标，需提高 DE_TARGET_MW 重跑")
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
        "x": np.asarray(best_x),
    }
    np.save(os.path.join(OUTPUT_DIR, "q2_result.npy"), summary)
    np.save(os.path.join(OUTPUT_DIR, "q2_sel.npy"), sel)
    print("  结果已缓存 → 02_论文/files/q2_result.npy, q2_sel.npy")
    return summary


if __name__ == "__main__":
    main()