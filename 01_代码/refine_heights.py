"""安装高度局部精调（坐标轮换）：固定塔位/尺寸/镜位，
逐带扫描 a ∈ {2.0, 2.5, ..., 6.0} 档位，低精度粗扫 + 高精度验证。

用法: OMP_NUM_THREADS=1 python3 refine_heights.py [--workers 16]
输出: 更新 q3_result.npy / q3_sel.npy，并重新导出 result3.xlsx、q3_表1-3.csv、布局图。
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "01_代码"))

from common import OUTPUT_DIR, FINAL_GRID, FINAL_RAYS
from q3_optimizer import (
    K_BANDS,
    REVIEW_GRID,
    REVIEW_RAYS,
    band_assign,
    export_all,
    field_eval,
)

RESULT_PATH = os.path.join(OUTPUT_DIR, "q3_result.npy")
SEL_PATH = os.path.join(OUTPUT_DIR, "q3_sel.npy")

A_LEVELS = np.arange(2.0, 6.05, 0.5)
IMPROVE_MIN = 0.01
MAX_ROUNDS = 3
GRID, RAYS = 6, 32


def eval_a(sel, tower, w, H, a, band, k, lev, grid_n, n_rays):
    a_try = a.copy()
    a_try[band == k] = lev
    P, eff = field_eval(sel, tower, w, H, a_try, grid_n, n_rays)
    return k, lev, float(P), eff


def refine_heights(sel, tower, w, H, a, band, workers=16,
                   grid_n=GRID, n_rays=RAYS, max_rounds=MAX_ROUNDS):
    """坐标轮换：逐带扫描安装高度档位，接受功率提升的改动，直至无改进。"""
    from multiprocessing import Pool

    a_new = a.copy()
    P_best, _ = field_eval(sel, tower, w, H, a_new, grid_n, n_rays)
    print(f"基准（低精度 {grid_n}×{grid_n}/{n_rays}）: P = {P_best:.3f} MW, "
          f"a = {[float(np.unique(a_new[band == k])[0]) for k in range(K_BANDS)]}",
          flush=True)

    for rnd in range(1, max_rounds + 1):
        improved = False
        for k in range(K_BANDS):
            cur = float(np.unique(a_new[band == k])[0])
            Hk = float(np.unique(H[band == k])[0])
            lo = float(np.ceil(Hk / 2.0 * 2) / 2)
            levels = np.unique(np.concatenate(
                ([cur], A_LEVELS[A_LEVELS >= lo - 1e-9])))
            tasks = [(sel, tower, w, H, a_new, band, k, lev, grid_n, n_rays)
                     for lev in levels if abs(lev - cur) > 1e-9]
            with Pool(workers) as pool:
                results = pool.starmap(eval_a, tasks)
            best_k, best_lev, best_P, best_eff = max(
                results, key=lambda r: r[2])
            if best_P > P_best + IMPROVE_MIN:
                a_new[band == best_k] = best_lev
                P_best, best_eff = best_P, best_eff
                improved = True
                print(f"  第 {rnd} 轮: 带 {best_k + 1} a {cur:.2f} → {best_lev:.1f} "
                      f"(P {P_best:.3f} MW)", flush=True)
            else:
                print(f"  第 {rnd} 轮: 带 {k + 1} a {cur:.2f} 保持 "
                      f"(最优档位 {best_lev:.1f}, P {best_P:.3f} MW)", flush=True)
        if not improved:
            break
    return a_new, P_best


def main():
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 16

    summary = np.load(RESULT_PATH, allow_pickle=True).item()
    sel = np.load(SEL_PATH)
    tower, w, H, a = summary["tower"], summary["w"], summary["H"], summary["a"]
    band = summary["band"]

    print(f"方案: 塔({tower[0]:.1f},{tower[1]:.1f}), N={len(sel)}, "
          f"workers={workers}, 精度 {GRID}×{GRID}/{RAYS}", flush=True)

    a_new, P_low = refine_heights(sel, tower, w, H, a, band, workers=workers)

    print("\n低精度最优组合验证（8×8/512）...", flush=True)
    P_mid, eff_mid = field_eval(sel, tower, w, H, a_new, REVIEW_GRID, REVIEW_RAYS)
    print(f"  8×8/512: P = {P_mid:.3f} MW", flush=True)

    print("最终高精度评价（10×10/1024）...", flush=True)
    P_final, eff = field_eval(sel, tower, w, H, a_new, FINAL_GRID, FINAL_RAYS)
    A_total = float(np.sum(w * H))
    q = P_final * 1000.0 / A_total
    print(f"\n======== 精调后最终结果 ========", flush=True)
    print(f"  吸收塔位置: ({tower[0]:.1f}, {tower[1]:.1f}) m", flush=True)
    print(f"  定日镜数: {len(sel)}, 镜面总面积: {A_total:.1f} m²", flush=True)
    print(f"  年平均输出热功率: {P_final:.3f} MW", flush=True)
    print(f"  单位镜面面积年平均输出热功率: {q:.4f} kW/m²", flush=True)
    print(f"  年均分项效率: 余弦 {eff['eta_cos']:.4f}, 阴影遮挡 {eff['eta_sb']:.4f}, "
          f"截断 {eff['eta_trunc']:.4f}, 综合 {eff['eta']:.4f}", flush=True)
    for k in range(K_BANDS):
        m = band == k
        print(f"  带 {k + 1}: {m.sum():4d} 面, a = {float(np.unique(a_new[m])[0]):.2f} m, "
              f"r∈[{np.hypot(sel[m,0]-tower[0], sel[m,1]-tower[1]).min():.1f},"
              f"{np.hypot(sel[m,0]-tower[0], sel[m,1]-tower[1]).max():.1f}] m", flush=True)

    summary["a"] = a_new
    summary["P_mw"] = P_final
    summary["q"] = q
    for k in ("eta_cos", "eta_sb", "eta_trunc", "eta"):
        summary[k] = eff[k]
    summary["x"][10] = float(np.unique(a_new[band == K_BANDS - 1])[0])
    np.save(RESULT_PATH, summary)
    np.save(SEL_PATH, sel)
    print("已更新 q3_result.npy / q3_sel.npy，重新导出全部产物...", flush=True)
    export_all(summary, sel)


if __name__ == "__main__":
    main()