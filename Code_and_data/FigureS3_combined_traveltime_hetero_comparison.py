# %% [markdown]
# # 该文件用于生成走时对比整合图（基于combined_ref.py排版）

# %%
from dataFunc.statisticalProcess import *
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
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

marker_colors = {
    "L0": "#1f77b4",
    "L1": "#d62728",
    "L2": "#7f7f7f",
    "L3": "#1a55a3",
    "L4": "#6a3d9a",
}

markers = {
    "L0": "o",
    "L1": "v",
    "L2": "p",
    "L3": "X",
    "L4": "D",
}

level_labels = ["Lv. 1", "Lv. 2", "Lv. 3", "Lv. 4", "Lv. 5"]
level_keys = ["L0", "L1", "L2", "L3", "L4"]
level_sizes = [40, 40, 40, 40, 60]
level_linewidths = [0.5, 0.5, 0.5, 0.5, 0.8]


def classify_tt_levels(inv_ttdata_all, ref_ttdata_all, row=5, col=10):
    n_all = min(len(inv_ttdata_all), len(ref_ttdata_all))
    inv_tt_l = {f"L{i}": [] for i in range(5)}
    ref_tt_l = {f"L{i}": [] for i in range(5)}

    for i in range(n_all):
        for r in range(row):
            for c in range(col):
                mod = int(np.abs(r - (c % row)))
                if mod <= 4:
                    inv_tt_l[f"L{mod}"].append(inv_ttdata_all[i][r][c])
                    ref_tt_l[f"L{mod}"].append(ref_ttdata_all[i][r][c])

    invs_all_list = []
    refs_all_list = []
    result = {}
    for lvl in range(5):
        key = f"L{lvl}"
        inv_arr = np.array(inv_tt_l[key])
        ref_arr = np.array(ref_tt_l[key])
        result[key] = (inv_arr, ref_arr)
        invs_all_list.append(inv_arr)
        refs_all_list.append(ref_arr)

    result["all"] = (np.concatenate(invs_all_list), np.concatenate(refs_all_list))
    return result


def plot_tt_subplot(ax, tt_data, describe, fig_letter):
    invs_all, refs_all = tt_data["all"]

    max_t = max(np.max(invs_all), np.max(refs_all))
    x_min = 0
    x_max = max_t + 0.2 * max_t + 1.5
    y_min = 0
    y_max = max_t + 0.2 * max_t + 1.5
    xlim = [x_min, x_max]
    ylim = [y_min, y_max]

    edgec = "w"
    for lvl_key, label, s, lw in zip(
        level_keys, level_labels, level_sizes, level_linewidths
    ):
        inv_data, ref_data = tt_data[lvl_key]
        ax.scatter(
            ref_data,
            inv_data,
            alpha=0.7,
            s=s,
            c=marker_colors[lvl_key],
            marker=markers[lvl_key],
            edgecolors=edgec,
            linewidths=lw,
        )

    ax.plot(xlim, ylim, "k--", linewidth=1.5, alpha=0.7)
    z_all = np.polyfit(refs_all, invs_all, 1)
    p = np.poly1d(z_all)
    ax.plot(xlim, [p(x) for x in xlim], "r-", linewidth=1.5)

    invs_all_fit = p(refs_all)
    r2_all = r2_score(invs_all, invs_all_fit)
    mse_all = mean_squared_error(refs_all, invs_all)
    rmse_all = np.sqrt(mse_all)
    mbe_all = np.mean(invs_all - refs_all)

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
        + rf"$MBE$: {mbe_all:.2f} $(s)$"
        + "\n"
        + rf"$RMSE$: {rmse_all:.2f} $(s)$"
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

    ax.text(
        0.02,
        0.98,
        f"({fig_letter})",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=18,
    )

    ax.text(
        0.95,
        0.05,
        describe,
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.8
        ),
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")


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

subplot_positions = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
]

for idx, sce_case in enumerate(scenarios_cases):
    scenario = sce_case["scenario"]
    case_name = sce_case["case"]
    describe = sce_case["describe"]
    fig_letter = sce_case["fig_index"]

    invpath = (
        f"{workpath}/{field_name}/{scenario}/{case_name}/model_export_inv"
    )
    refpath = f"{workpath}/{field_name}/{scenario}/{case_name}/model_export"

    ttdata_inv = read_correct_ttdata(invpath)
    ttdata_ref = read_correct_ttdata(refpath)

    inv_ttdata_all = np.array(ttdata_inv.correct_data)
    ref_ttdata_all = np.array(ttdata_ref.correct_data)

    tt_data = classify_tt_levels(inv_ttdata_all, ref_ttdata_all)

    row, col = subplot_positions[idx]
    ax = axes[row][col]
    plot_tt_subplot(ax, tt_data, describe, fig_letter)

    if col == 0:
        ax.set_ylabel(r"Estimated Travel Time ($s$)")
    if row == rows - 1:
        ax.set_xlabel(r"Reference Travel Time ($s$)")

for col in [2, 3]:
    axes[0][col].axis("off")

legend_ax = axes[0][2]
legend_ax.axis("on")
legend_ax.set_xticks([])
legend_ax.set_yticks([])
for spine in legend_ax.spines.values():
    spine.set_visible(False)

legend_elements = []
for lvl_key, label, s in zip(level_keys, level_labels, level_sizes):
    legend_elements.append(
        mlines.Line2D(
            [],
            [],
            color="w",
            marker=markers[lvl_key],
            markerfacecolor=marker_colors[lvl_key],
            markeredgecolor="w",
            linestyle="None",
            markersize=10,
            label=label,
            markeredgewidth=0.5,
        )
    )

legend_elements.append(
    mlines.Line2D([], [], color="black", linestyle="--", linewidth=2, label=r"$y=x$")
)
legend_elements.append(
    mlines.Line2D([], [], color="red", linestyle="-", linewidth=2, label="Fitted line")
)

legend_ax.legend(
    handles=legend_elements,
    loc="upper left",
    fontsize=16,
    framealpha=0.9,
    edgecolor="gray",
)

plt.tight_layout()
plt.subplots_adjust(top=0.92)

savepath = "./figure_table_output/Figure/FigureS3"
import os

os.makedirs(savepath, exist_ok=True)
plt.savefig(
    f"{savepath}/FigureS3_combined_ref_est_traveltime.png", dpi=300, bbox_inches="tight"
)
plt.savefig(
    f"{savepath}/FigureS3_combined_ref_est_traveltime.svg",
    format="svg",
    bbox_inches="tight",
)
plt.show()
