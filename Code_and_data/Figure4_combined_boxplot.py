import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataFunc.statisticalProcess import read_refdata
from sklearn.metrics import mean_squared_error

from plotFunc.boxplot import (
    setup_boxplot_style,
    boxplot_2case,
    boxplot_4case,
    boxplot_case_1a2a,
    create_boxplot_legend,
)
from geostatFunc.statistics import setup_case_2a, setup_case_2bc

# ============================================================
#  1.  Configure scenarios, cases, output
# ============================================================
field_name = "dataset_scenarios"
language = "en"
save_dir = "./figure_table_output/Figure/Figure4"

import os

os.makedirs(save_dir, exist_ok=True)

setup_boxplot_style(language)

data_label = ["RMSE", "MBE", "Pearson's r", "Slope"]

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

scenario_case = [
    {"scenario": scenarios[0], "case": cases_name[0]},
    {"scenario": scenarios[0], "case": cases_name[1]},
    # 2b case1-4
    {"scenario": scenarios[0], "case": cases_name[1]},
    {"scenario": scenarios[1], "case": cases_name[2]},
    {"scenario": scenarios[1], "case": cases_name[3]},
    {"scenario": scenarios[1], "case": cases_name[4]},
    # 2c case1-4
    {"scenario": scenarios[2], "case": cases_name[5]},
    {"scenario": scenarios[0], "case": cases_name[1]},
    {"scenario": scenarios[2], "case": cases_name[6]},
    {"scenario": scenarios[2], "case": cases_name[7]},
]

ax_lim_2a = {
    "ax1": [0, 1.5],
    "ax2": [-0.3, 0.2],
    "ax3": [0.5, 1.0],
    "ax4": [0.2, 0.7],
}
ax_lim_2b = {
    "ax1": [0.5, 3],
    "ax2": [-0.7, 0.3],
    "ax3": [0.5, 1.0],
    "ax4": [0.1, 0.6],
}
ax_lim_2c = {
    "ax1": [0.5, 1],
    "ax2": [-0.5, 0],
    "ax3": [0.5, 1.0],
    "ax4": [0.1, 0.6],
}

# ============================================================
#  2.  Load & compute data
# ============================================================
print("Loading data...")

case_data_2a = [
    {data_label[0]: [], data_label[1]: [], data_label[2]: [], data_label[3]: []},
    {data_label[0]: [], data_label[1]: [], data_label[2]: [], data_label[3]: []},
]
case_data_2b = [
    {data_label[0]: [], data_label[1]: [], data_label[2]: [], data_label[3]: []},
    {data_label[0]: [], data_label[1]: [], data_label[2]: [], data_label[3]: []},
    {data_label[0]: [], data_label[1]: [], data_label[2]: [], data_label[3]: []},
    {data_label[0]: [], data_label[1]: [], data_label[2]: [], data_label[3]: []},
]
case_data_2c = [
    {data_label[0]: [], data_label[1]: [], data_label[2]: [], data_label[3]: []},
    {data_label[0]: [], data_label[1]: [], data_label[2]: [], data_label[3]: []},
    {data_label[0]: [], data_label[1]: [], data_label[2]: [], data_label[3]: []},
    {data_label[0]: [], data_label[1]: [], data_label[2]: [], data_label[3]: []},
]

setup_case_2a(case_data_2a, data_label, scenario_case[0:2], field_name)
setup_case_2bc(case_data_2b, data_label, scenario_case[2:6], field_name)
setup_case_2bc(case_data_2c, data_label, scenario_case[6:10], field_name)

# Load 1a homogeneous data for scatter overlay (if available)
field_1a = "dataset_scenarios"
scen_1a = "1a"
cases_1a = ["mean_-11.5", "mean_-9.2"]

case_data_1a = None
if os.path.isdir(f"./data/{field_1a}"):
    rmse1a, mbe1a = [], []
    for case_n in cases_1a:
        folder = f"./data/{field_1a}/{scen_1a}/{case_n}/lnKs"
        homo_est = read_refdata(f"{folder}/invsResult")
        homo_ref = read_refdata(f"{folder}/homo_field")
        est = np.array(homo_est.data) + np.log(6)
        ref = np.array(homo_ref.data)
        e, r = est[0], ref[0]
        rmse1a.append(np.sqrt(mean_squared_error(r, e)))
        mbe1a.append(np.mean(e - r))
    case_data_1a = [
        {data_label[0]: [rmse1a[0]], data_label[1]: [mbe1a[0]]},
        {data_label[0]: [rmse1a[1]], data_label[1]: [mbe1a[1]]},
    ]
else:
    print("1a field data not found, skipping scatter overlay.")

print("Data loaded.")

# ============================================================
#  3.  Save individual subfigures
# ============================================================
print("Saving individual subfigures...")

# --- 2a (2-case) ---
fig, ax = plt.subplots(figsize=(16, 5))
boxplot_2case(
    ax,
    case_data_2a,
    data_label,
    scenario_name="2a",
    ax_lim=ax_lim_2a,
    language=language,
)
fig.tight_layout()
fig.savefig(f"{save_dir}/Scenario_2a_metrics_boxplot.png", dpi=300, bbox_inches="tight")
fig.savefig(
    f"{save_dir}/Scenario_2a_metrics_boxplot.svg",
    format="svg",
    bbox_inches="tight",
    pad_inches=0.1,
    transparent=True,
)
plt.close(fig)

# --- 2b (4-case) ---
fig, ax = plt.subplots(figsize=(16, 5))
boxplot_4case(
    ax,
    case_data_2b,
    data_label,
    scenario_name="2b",
    ax_lim=ax_lim_2b,
    language=language,
)
fig.tight_layout()
fig.savefig(f"{save_dir}/Scenario_2b_metrics_boxplot.png", dpi=300, bbox_inches="tight")
fig.savefig(
    f"{save_dir}/Scenario_2b_metrics_boxplot.svg",
    format="svg",
    bbox_inches="tight",
    pad_inches=0.1,
    transparent=True,
)
plt.close(fig)

# --- 2c (4-case) ---
fig, ax = plt.subplots(figsize=(16, 5))
boxplot_4case(
    ax,
    case_data_2c,
    data_label,
    scenario_name="2c",
    ax_lim=ax_lim_2c,
    language=language,
)
fig.tight_layout()
fig.savefig(f"{save_dir}/Scenario_2c_metrics_boxplot.png", dpi=300, bbox_inches="tight")
fig.savefig(
    f"{save_dir}/Scenario_2c_metrics_boxplot.svg",
    format="svg",
    bbox_inches="tight",
    pad_inches=0.1,
    transparent=True,
)
plt.close(fig)

# --- 1a + 2a ---
fig, ax = plt.subplots(figsize=(16, 5))
boxplot_case_1a2a(
    ax,
    case_data_2a,
    data_label,
    case_data_1a=case_data_1a,
    scenario_name="1a and 2a",
    ax_lim=ax_lim_2a,
    language=language,
)
fig.tight_layout()
fig.savefig(
    f"{save_dir}/Scenario_1a_and_2a_metrics_boxplot.png", dpi=300, bbox_inches="tight"
)
fig.savefig(
    f"{save_dir}/Scenario_1a_and_2a_metrics_boxplot.svg",
    format="svg",
    bbox_inches="tight",
    pad_inches=0.1,
    transparent=True,
)
plt.close(fig)

print("Individual subfigures saved.")

# ============================================================
#  4.  Combined figure: 3 rows (1a2a, 2b, 2c) + legend below
# ============================================================
print("Creating combined figure...")

fig_combined = plt.figure(figsize=(16, 16))
gs = GridSpec(
    4, 1, figure=fig_combined, height_ratios=[1.0, 1.0, 1.0, 0.12], hspace=0.45
)

ax_row1 = fig_combined.add_subplot(gs[0, 0])
ax_row2 = fig_combined.add_subplot(gs[1, 0])
ax_row3 = fig_combined.add_subplot(gs[2, 0])
ax_legend = fig_combined.add_subplot(gs[3, 0])

# Row 1: 1a + 2a
boxplot_case_1a2a(
    ax_row1,
    case_data_2a,
    data_label,
    case_data_1a=case_data_1a,
    scenario_name="1a and 2a",
    ax_lim=ax_lim_2a,
    language=language,
    title="(a) scenario 1a and 2a",
)

# Row 2: 2b
boxplot_4case(
    ax_row2,
    case_data_2b,
    data_label,
    scenario_name="2b",
    ax_lim=ax_lim_2b,
    language=language,
    title="(b) scenario 2b",
)

# Row 3: 2c
boxplot_4case(
    ax_row3,
    case_data_2c,
    data_label,
    scenario_name="2c",
    ax_lim=ax_lim_2c,
    language=language,
    title="(c) scenario 2c",
)

# Legend below
create_boxplot_legend(ax_legend, language=language)

fig_combined.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.04, hspace=0.50)
fig_combined.savefig(
    f"{save_dir}/Figure4_combined_boxplot.png", dpi=300, bbox_inches="tight"
)
fig_combined.savefig(
    f"{save_dir}/Figure4_combined_boxplot.svg",
    format="svg",
    bbox_inches="tight",
    pad_inches=0.1,
    transparent=True,
)
print(f"Combined figure saved to {save_dir}/Figure4_combined_boxplot.png")
print("Done!")
