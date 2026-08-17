"""生成论文研究技术路线流程图（三层框架）。

输出：02_论文/figures/fig1_framework.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

C1 = "#2c7fb8"
C2 = "#31a354"
C3 = "#de2d26"
C4 = "#756bb1"
GRAY = "#555555"


def box(x, y, w, h, text, fc, ec, fs=9.5):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       fc=fc, ec=ec, lw=1.4)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color="white", wrap=True)
    return (x, y, w, h)


def arrow(p1, p2, color=GRAY, style="-|>"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                        color=color, lw=1.3)
    ax.add_patch(a)


# ---- 第一层：问题一（口径基准）----
box(0.4, 5.0, 9.2, 0.85,
    "问题一：五分量光学效率追迹模型（60 时刻 / 射线-矩形、射线-圆柱求交）\n"
    "年平均光学效率 0.5777 · 年平均功率 35.32 MW · 单位面积功率 0.5622 kW/m²",
    "#2c7fb8", "#1f5f8f", fs=9)

# ---- 第二层：问题二（分层优化）----
box(0.4, 3.0, 2.15, 1.5,
    "外层\n差分进化(DE)\n塔位/尺寸/安装\n高度/晶格参数\n8 维全局搜索", C2, "#23703c")
box(2.85, 3.0, 2.15, 1.5,
    "内层\n六角晶格\n候选镜位\n快速效率地图\n贪心选镜", C2, "#23703c")
box(5.3, 3.0, 2.15, 1.5,
    "约束检查\n场地圆 / 塔周禁装\n最小间距 w+5\n功率 ≥ 60 MW", C4, "#51418a")
box(7.75, 3.0, 1.85, 1.5,
    "复核层\n完整 60 时刻\n射线追迹\n精确复算", "#de2d26", "#a02020")

arrow((2.55, 3.75), (2.85, 3.75))
arrow((5.0, 3.75), (5.3, 3.75))
arrow((7.45, 3.75), (7.75, 3.75))
arrow((8.67, 3.0), (8.67, 2.35), color=C3)
arrow((2.5, 3.0), (2.5, 2.35), color=C2)

# ---- 反馈回路文字 ----
ax.text(5.0, 2.62, "局部改进：塔位扰动 / 尺寸邻域 / 单镜移位删补 / 一换一替换",
        ha="center", va="center", fontsize=8.5, color=GRAY)

# ---- 第三层：问题三（扩展）----
box(0.4, 0.55, 4.4, 1.0,
    "问题三扩展：逐镜尺寸与安装高度档位离散化\n"
    "内层按\"单位面积贡献\"逐镜独立选档", "#31a354", "#23703c")
box(5.3, 0.55, 4.3, 1.0,
    "结果交付：result2.xlsx / result3.xlsx\n表 1·表 2·表 3 · 布局图 · 优化日志", "#756bb1", "#51418a")

arrow((4.8, 1.05), (5.3, 1.05))
arrow((5.0, 3.0), (5.0, 1.55), color=C4, style="<|-|>")
arrow((2.6, 0.55), (2.6, 0.15), color=GRAY)
arrow((6.95, 0.15), (6.95, 0.55), color=GRAY)
ax.text(4.75, 0.2, "同一追迹口径 + 复核验证衔接", fontsize=8, color=GRAY,
        ha="center")

plt.tight_layout()
out = "/home/gsh/Mathematical Modeling/02_论文/figures/fig1_framework.png"
plt.savefig(out, bbox_inches="tight", facecolor="white")
print("已生成:", out)