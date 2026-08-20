#!/usr/bin/env python3
"""边际镜删除：删除年均净贡献最低的镜子，在保持 P̄≥60 MW 下提升 q。

流程：
1. 读 q3_result.npy / q3_sel.npy；
2. 低精度 6×6/32 全场评价（return_per_mirror）取每镜年均功率；
3. 迭代删除：每轮删贡献最低的 n_drop 面，重新评价，直到低精度 P ≤ 60.35；
4. 8×8/512 复核 + 10×10/1024 终验（≥60 才接受）；
5. 若 q 提升 > 0.3%，更新 summary 并 export_all 重导出。
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q2_optimizer as q2
import q3_optimizer as q3

OUTPUT_DIR = q2.OUTPUT_DIR
RESULT_PATH = os.path.join(OUTPUT_DIR, "q3_result.npy")
SEL_PATH = os.path.join(OUTPUT_DIR, "q3_sel.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--target", type=float, default=60.55,
                    help="低精度停止功率（留复核裕量）")
    ap.add_argument("--n-drop", type=int, default=15, help="每轮删除镜数")
    ap.add_argument("--n-max", type=int, default=0,
                    help="最大删除总数（0=不限，由低精度目标决定）")
    ap.add_argument("--improve-min", type=float, default=0.003,
                    help="q 提升阈值（kW/m²），低于则不更新")
    args = ap.parse_args()

    summary = np.load(RESULT_PATH, allow_pickle=True).item()
    sel = np.load(SEL_PATH)
    tower = summary["tower"]
    w_arr = np.asarray(summary["w"], float)
    H_arr = np.asarray(summary["H"], float)
    a_arr = np.asarray(summary["a"], float)
    print(f"初始: N={len(sel)}, A={np.sum(w_arr*H_arr):.1f}, "
          f"P={summary['P_mw']:.3f} MW, q={summary['q']:.4f}")

    keep = np.ones(len(sel), dtype=bool)
    P_est, _, pm = q2.field_eval(sel, tower, w_arr, H_arr, a_arr, 6, 32,
                                 return_per_mirror=True)
    print(f"低精度基准: P={P_est:.3f} MW (每镜平均 {pm.mean():.2f} kW)")
    n_round = 0
    while P_est > args.target and (args.n_max <= 0 or keep.sum() > len(sel) - args.n_max):
        n_round += 1
        order = np.argsort(pm)
        drop = order[:args.n_drop]
        keep[drop] = False
        sel_k = sel[keep]
        w_k, H_k, a_k = w_arr[keep], H_arr[keep], a_arr[keep]
        P_est, _, pm = q2.field_eval(sel_k, tower, w_k, H_k, a_k, 6, 32,
                                     return_per_mirror=True)
        removed = len(sel) - len(sel_k)
        A_k = np.sum(w_k * H_k)
        print(f"  第 {n_round} 轮: 累计删 {removed} 面, N={len(sel_k)}, "
              f"A={A_k:.1f}, P={P_est:.3f} MW, q={P_est*1000/A_k:.4f}")
    print(f"删除结束: 共删 {len(sel)-len(sel_k)} 面, 低精度 P={P_est:.3f} MW")

    P_r, eff_r = q2.field_eval(sel_k, tower, w_k, H_k, a_k, 8, 512)
    print(f"8×8/512 复核: P={P_r:.3f} MW")
    if P_r < 60.0:
        print("复核未达标，回退：不删除任何镜，维持原方案")
        return
    P_f, eff = q2.field_eval(sel_k, tower, w_k, H_k, a_k, 10, 1024)
    print(f"10×10/1024 终验: P={P_f:.3f} MW, q={P_f*1000/np.sum(w_k*H_k):.4f}")
    if P_f < 60.0:
        print("终验未达标，回退：维持原方案")
        return

    A_new = float(np.sum(w_k * H_k))
    q_new = P_f * 1000.0 / A_new
    q_old = summary["q"]
    if q_new - q_old < args.improve_min:
        print(f"q 提升 {q_new - q_old:+.4f} < 阈值 {args.improve_min}，维持原方案")
        return

    print(f"接受删除方案: N={len(sel_k)}, A={A_new:.1f}, P={P_f:.3f} MW, "
          f"q={q_new:.4f} (+{q_new-q_old:.4f})")
    summary["N"] = len(sel_k)
    summary["A_total"] = A_new
    summary["P_mw"] = P_f
    summary["q"] = q_new
    for k, v in eff.items():
        summary[k] = v
    summary["w"] = w_k
    summary["H"] = H_k
    summary["a"] = a_k
    band_id = q3.band_assign(sel_k, tower)
    summary["band"] = band_id
    np.save(RESULT_PATH, summary)
    np.save(SEL_PATH, sel_k)
    q3.export_all(summary, sel_k)
    print("q3_result.npy / q3_sel.npy / 全部导出已更新")


if __name__ == "__main__":
    main()