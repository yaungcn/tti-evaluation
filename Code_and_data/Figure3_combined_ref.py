# %% [markdown]
# # 该文件用于统计对比 TrueD 与 EstD，并生成组合图

# %%
from dataFunc.statisticalProcess import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import r2_score, mean_squared_error

font = "Arial"
math_font = "stixsans"
workpath = "./data"
field_name = "dataset_scenarios"
scenarios = ["2a", "2b", "2c"]
cases_name = [
    "mean_exp_-11.5",
    "mean_exp_-9.2",
    "var_lnK_2",
    "var_lnK_4",
    "var_lnK_8",
    "cor_2",
    "cor_6",
    "cor_8",
]
cases_idx = [1, 2, 2, 3, 4, 1, 3, 4]

scenarios_cases = [
    {
        "scenario": scenarios[0],
        "case": cases_name[0],
        "case_index": 1,
        "fig_index": "a",
        "describe": "scenario 2a, case1: \nmean ln(K) = -11.5",
    },
    {
        "scenario": scenarios[0],
        "case": cases_name[1],
        "case_index": 2,
        "fig_index": "b",
        "describe": "scenario 2a, case2: \nmean ln(K) = -9.2",
    },
    {
        "scenario": scenarios[0],
        "case": cases_name[1],
        "case_index": 1,
        "fig_index": "c",
        "describe": "scenario 2b, case1: \nvar. ln(K) = 1",
    },
    {
        "scenario": scenarios[1],
        "case": cases_name[2],
        "case_index": 2,
        "fig_index": "d",
        "describe": "scenario 2b, case2: \nvar. ln(K) = 2",
    },
    {
        "scenario": scenarios[1],
        "case": cases_name[3],
        "case_index": 3,
        "fig_index": "e",
        "describe": "scenario 2b, case3: \nvar. ln(K) = 4",
    },
    {
        "scenario": scenarios[1],
        "case": cases_name[4],
        "case_index": 4,
        "fig_index": "f",
        "describe": "scenario 2b, case4: \nvar. ln(K) = 8",
    },
    {
        "scenario": scenarios[2],
        "case": cases_name[5],
        "case_index": 1,
        "fig_index": "g",
        "describe": "scenario 2b, case1: \ncor. ln(K) = 2",
    },
    {
        "scenario": scenarios[0],
        "case": cases_name[1],
        "case_index": 2,
        "fig_index": "h",
        "describe": "scenario 2b, case2: \ncor. ln(K) = 4",
    },
    {
        "scenario": scenarios[2],
        "case": cases_name[6],
        "case_index": 3,
        "fig_index": "i",
        "describe": "scenario 2b, case3: \ncor. ln(K) = 6",
    },
    {
        "scenario": scenarios[2],
        "case": cases_name[7],
        "case_index": 4,
        "fig_index": "j",
        "describe": "scenario 2b, case4: \ncor. ln(K) = 8",
    },
]


# 定义子图绘制函数（复用原代码逻辑）
def plot_subplot(ax, ref_D, inv_D, param_name, describe, fig_letter, unit="m/s"):
    # 准备数据
    n_all = ref_D.shape[0]
    invs_all = []
    refs_all = []
    for i in range(n_all):
        invs_all.append(inv_D[i].flatten().tolist())
        refs_all.append(ref_D[i].flatten().tolist())
    invs_all = np.array(invs_all).flatten()
    refs_all = np.array(refs_all).flatten()

    max_D_all = max(np.max(invs_all), np.max(refs_all))
    min_D_all = min(np.min(invs_all), np.min(refs_all))
    xlim_all = [int(min_D_all) - 1.5, int(max_D_all) + 1.5]
    ylim_all = [int(min_D_all) - 1.5, int(max_D_all) + 1.5]

    # 绘制散点
    ax.scatter(refs_all, invs_all, marker="+", alpha=0.8, s=80, c="k")

    # 45度参考线
    ax.plot(xlim_all, ylim_all, color="black", linestyle="--")

    # 拟合线
    z_all = np.polyfit(refs_all, invs_all, 1)
    p = np.poly1d(z_all)
    ax.plot(xlim_all, [p(x) for x in xlim_all], color="red", linestyle="-")

    # 坐标轴设置
    ax.set_xlim(xlim_all)
    ax.set_ylim(ylim_all)
    ax.set_aspect("equal")

    # 计算指标
    invs_all_fit = p(refs_all)
    r2_all = r2_score(invs_all, invs_all_fit)
    mse_all = mean_squared_error(refs_all, invs_all)
    rmse_all = np.sqrt(mse_all)
    mbe_all = np.mean(invs_all - refs_all)

    # 文本内容（左上角）
    if z_all[1] >= 0:
        text_func = (
            r"$y = $" + f"{z_all[0]:.2f}" + r"$x + $" + f"{np.abs(z_all[1]):.2f}\n"
        )
    else:
        text_func = (
            r"$y = $" + f"{z_all[0]:.2f}" + r"$x - $" + f"{np.abs(z_all[1]):.2f}\n"
        )
    text_content = (
        text_func
        + rf"$R^2$: {r2_all:.2f}"
        + "\n"
        + rf"$MBE$: {mbe_all:.2f}"
        + "\n"
        + rf"$RMSE$: {rmse_all:.2f}"
    )
    ax.text(
        0.15,
        0.95,
        text_content,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.8
        ),
        fontsize=14,
    )

    # 子图字母 (左上角)
    ax.text(
        0.02,
        0.98,
        f"({fig_letter})",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=18,
        # fontweight="bold",
    )

    # 描述文字 (右下角)

    ax.text(
        0.50,
        0.12,
        describe,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="center",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.8
        ),
    )


# 主循环：收集所有数据并绘制组合图
import matplotlib.lines as mlines

# 准备画布：3行4列
rows = 3
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(20, 15))

plt.rcParams["font.family"] = font
plt.rcParams["mathtext.fontset"] = math_font
plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
)

# 定义子图在网格中的位置 (row, col)
# 索引 0 → (0,0), 1 → (0,1)
# 索引 2 → (1,0), 3 → (1,1), 4 → (1,2), 5 → (1,3)
# 索引 6 → (2,0), 7 → (2,1), 8 → (2,2), 9 → (2,3)
subplot_positions = [
    (0, 0),
    (0, 1),  # 第一行 a, b
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),  # 第二行 c, d, e, f
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),  # 第三行 g, h, i, j
]

# 遍历每个场景，读取数据并绘图
for idx, sce_case in enumerate(scenarios_cases):
    scenario = sce_case["scenario"]
    case_name = sce_case["case"]
    describe = sce_case["describe"]
    fig_letter = sce_case["fig_index"]

    # 读取数据
    invdata_lnKs = read_refdata(
        f"{workpath}/{field_name}/{scenario}/{case_name}/lnKs/invsResult"
    )
    refdata_lnKs = read_refdata(
        f"{workpath}/{field_name}/{scenario}/{case_name}/lnKs/hetero_field"
    )
    ref_Ks = np.array(refdata_lnKs.data)
    inv_Ks = np.array(invdata_lnKs.data)

    # 根据预定义的位置获取对应的axes
    row, col = subplot_positions[idx]
    ax = axes[row][col]
    plot_subplot(ax, ref_Ks, inv_Ks, "ln(K)", describe, fig_letter)

    # 坐标轴标签只加在左边和底部的子图上
    if col == 0:
        ax.set_ylabel(r"Estimated $ln(K)$")
    if row == rows - 1:  # 最后一行
        ax.set_xlabel(r"Reference $ln(K)$")

# 第一行的后两列（(0,2) 和 (0,3)）始终隐藏
for col in [2, 3]:
    axes[0][col].axis("off")

# 在第一行第三列（axes[0][2]）添加图例
legend_ax = axes[0][2]
legend_ax.axis("on")  # 开启坐标轴以显示图例
legend_ax.set_xticks([])
legend_ax.set_yticks([])
# 移除边框（可选）
for spine in legend_ax.spines.values():
    spine.set_visible(False)

legend_elements = [
    mlines.Line2D(
        [],
        [],
        color="k",
        marker="+",
        linestyle="None",
        markersize=10,
        label=r"Est. vs Ref. $ln(K)$",
    ),
    mlines.Line2D([], [], color="black", linestyle="--", linewidth=2, label=r"$y=x$"),
    mlines.Line2D([], [], color="red", linestyle="-", linewidth=2, label="Fitted line"),
]

legend_ax.legend(
    handles=legend_elements,
    loc="upper left",
    fontsize=16,
    framealpha=0.9,
    edgecolor="gray",
)

# # 添加统一图例（右上角）
# legend_elements = [
#     mlines.Line2D(
#         [],
#         [],
#         color="k",
#         marker="+",
#         linestyle="None",
#         markersize=10,
#         label="Est. vs Ref. ln(K)",
#     ),
#     mlines.Line2D([], [], color="black", linestyle="--", linewidth=2, label=r"$y=x$"),
#     mlines.Line2D([], [], color="red", linestyle="-", linewidth=2, label="Fitted line"),
# ]
# fig.legend(
#     handles=legend_elements,
#     loc="upper right",
#     bbox_to_anchor=(0.98, 0.98),
#     fontsize=16,
#     framealpha=0.9,
# )

# 调整子图间距
plt.tight_layout()
plt.subplots_adjust(top=0.92)  # 为图例留出空间

# 保存图像
savepath = "./figure_table_output/Figure/Figure3"
import os

os.makedirs(savepath, exist_ok=True)
plt.savefig(
    f"{savepath}/Figure3_combined_ref_est_lnK.png", dpi=300, bbox_inches="tight"
)
plt.savefig(
    f"{savepath}/Figure3_combined_ref_est_lnK.svg", format="svg", bbox_inches="tight"
)
# plt.show()
