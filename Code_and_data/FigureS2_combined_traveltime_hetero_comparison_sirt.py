# %% [markdown]
# # 该文件用于生成 SIRT 反演结果的 eikonal 走时对比整合图
# # 依赖 compute_traveltime_sirt.py 预计算的数据

# %%
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
from sklearn.metrics import r2_score, mean_squared_error

font = "Arial"
math_font = "stixsans"

scenarios_cases = [
    {
        "scenario": "2a",
        "case": "mean_exp_-11.5",
        "case_index": 1,
        "fig_index": "a",
        "describe": "scenario 2a, case1: \nmean ln(K) = -11.5",
    },
    {
        "scenario": "2a",
        "case": "mean_exp_-9.2",
        "case_index": 2,
        "fig_index": "b",
        "describe": "scenario 2a, case2: \nmean ln(K) = -9.2",
    },
    {
        "scenario": "2a",
        "case": "mean_exp_-9.2",
        "case_index": 1,
        "fig_index": "c",
        "describe": "scenario 2b, case1: \nvar. ln(K) = 1",
    },
    {
        "scenario": "2b",
        "case": "var_lnK_2",
        "case_index": 2,
        "fig_index": "d",
        "describe": "scenario 2b, case2: \nvar. ln(K) = 2",
    },
    {
        "scenario": "2b",
        "case": "var_lnK_4",
        "case_index": 3,
        "fig_index": "e",
        "describe": "scenario 2b, case3: \nvar. ln(K) = 4",
    },
    {
        "scenario": "2b",
        "case": "var_lnK_8",
        "case_index": 4,
        "fig_index": "f",
        "describe": "scenario 2b, case4: \nvar. ln(K) = 8",
    },
    {
        "scenario": "2c",
        "case": "cor_2",
        "case_index": 1,
        "fig_index": "g",
        "describe": "scenario 2c, case1: \ncor. ln(K) = 2",
    },
    {
        "scenario": "2a",
        "case": "mean_exp_-9.2",
        "case_index": 2,
        "fig_index": "h",
        "describe": "scenario 2b, case2: \ncor. ln(K) = 4",
    },
    {
        "scenario": "2c",
        "case": "cor_6",
        "case_index": 3,
        "fig_index": "i",
        "describe": "scenario 2c, case3: \ncor. ln(K) = 6",
    },
    {
        "scenario": "2c",
        "case": "cor_8",
        "case_index": 4,
        "fig_index": "j",
        "describe": "scenario 2c, case4: \ncor. ln(K) = 8",
    },
]


def flatten_tt_data(inv_ttdata_all, ref_ttdata_all):
    invs_all = inv_ttdata_all.reshape(-1)
    refs_all = ref_ttdata_all.reshape(-1)
    return invs_all, refs_all


def plot_tt_subplot(ax, invs_all, refs_all, describe, fig_letter):
    max_t = max(np.max(invs_all), np.max(refs_all))
    min_t = min(np.min(invs_all), np.min(refs_all))
    xlim = [0, int(max_t) + 1.5]
    ylim = [0, int(max_t) + 1.5]

    ax.scatter(refs_all, invs_all, marker="+", alpha=0.8, s=80, c="k")

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


data_root = "./data/dataset_sirt_traveltime"

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

    data_dir = os.path.join(data_root, scenario, case_name)
    ref_file = os.path.join(data_dir, "ref_ttdata.npy")
    eikonal_file = os.path.join(data_dir, "eikonal_ttdata.npy")

    if not os.path.exists(ref_file) or not os.path.exists(eikonal_file):
        print(f"[{fig_letter}] Missing data for {scenario}/{case_name}, skipping.")
        continue

    print(f"[{fig_letter}] Loading {scenario}/{case_name} ...")
    ref_ttdata_all = np.stack(np.load(ref_file, allow_pickle=True))
    eikonal_ttdata_all = np.stack(np.load(eikonal_file, allow_pickle=True))

    invs_all, refs_all = flatten_tt_data(eikonal_ttdata_all, ref_ttdata_all)

    row, col = subplot_positions[idx]
    ax = axes[row][col]
    plot_tt_subplot(ax, invs_all, refs_all, describe, fig_letter)

    if col == 0:
        ax.set_ylabel(r"Eikonal Travel Time $(s)$")
    if row == rows - 1:
        ax.set_xlabel(r"Reference Travel Time $(s)$")

for col in [2, 3]:
    axes[0][col].axis("off")

legend_ax = axes[0][2]
legend_ax.axis("on")
legend_ax.set_xticks([])
legend_ax.set_yticks([])
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
        label=r"Est. vs Ref. Travel Time",
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

plt.tight_layout()
plt.subplots_adjust(top=0.92)

savepath = "./figure_table_output/Figure/FigureS2"
os.makedirs(savepath, exist_ok=True)
plt.savefig(
    f"{savepath}/FigureS2_combined_eikonal_ref_traveltime.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    f"{savepath}/FigureS2_combined_eikonal_ref_traveltime.svg",
    format="svg",
    bbox_inches="tight",
)
plt.show()
