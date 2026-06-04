import os
import numpy as np
import matplotlib.pyplot as plt
from dataFunc.statisticalProcess import read_correct_ttdata
from dataFunc.traveltimeProcess import classify_ttdata_by_level
from plotFunc.comparisonPlot import draw_est_ref_tt_comparison
from plotFunc.traveltimeScatter import (
    generate_all_consistent_plots,
    create_tt_level_legend,
    TABLEAU_COLORS,
    MARKERS,
)

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

scen_idx = 0
case_idx = 1
case_num = cases_idx[case_idx]
scenario = scenarios[scen_idx]
case_name = cases_name[case_idx]

# ============================================================
# Figure 5a: Eikonal travel time comparison (precomputed data)
# ============================================================
print("=" * 60)
print("Figure 5a: Eikonal travel time comparison")
print("=" * 60)

scenario_case = {
    "scenario": scenario,
    "case": case_name,
    "case_index": case_num,
    "fig_index": "a",
}

save_dir = os.path.join("./data/dataset_sirt_traveltime", scenario, case_name)
ref_file = os.path.join(save_dir, "ref_ttdata.npy")
eikonal_file = os.path.join(save_dir, "eikonal_ttdata.npy")

if os.path.exists(ref_file) and os.path.exists(eikonal_file):
    print(f"Loading precomputed data from {save_dir}")
    ref_ttdata = np.load(ref_file, allow_pickle=True)
    eikonal_ttdata = np.load(eikonal_file, allow_pickle=True)

    eikonal_savepath = "./figure_table_output/Figure/Figure5/eikonal-comparison"
    os.makedirs(eikonal_savepath, exist_ok=True)

    draw_est_ref_tt_comparison(
        scenario_case,
        ref_ttdata,
        eikonal_ttdata,
        param_name="travel time",
        unit="s",
        num_all_case=200,
        save=True,
        save_path=eikonal_savepath,
        show_fig_indx=False,
        text_xy=[0.55, 0.35],
    )
    print(f"[5a] Eikonal comparison figure saved to {eikonal_savepath}/")
else:
    print(f"[5a] Precomputed data not found at {save_dir}")
    print("    Please run FigureS2_step1_compute_traveltime_sirt.py first.")

# ============================================================
# Figure 5b-g: SIRT inversion travel time level comparison
# ============================================================
print("\n" + "=" * 60)
print("Figure 5b-g: SIRT travel time comparison by path level")
print("=" * 60)

invpath = f"{workpath}/{field_name}/{scenario}/{case_name}/model_export_inv"
refpath = f"{workpath}/{field_name}/{scenario}/{case_name}/model_export"

print(f"Reading SIRT ttdata from:\n  inv: {invpath}\n  ref: {refpath}")
ttdata_inv = read_correct_ttdata(invpath)
ttdata_ref = read_correct_ttdata(refpath)

inv_ttdata_all = np.array(ttdata_inv.correct_data)
ref_ttdata_all = np.array(ttdata_ref.correct_data)

classified = classify_ttdata_by_level(inv_ttdata_all, ref_ttdata_all, n_all=200)
inv_data = classified["inv"]
ref_data = classified["ref"]

n_all = 200
invs_all = []
refs_all = []
for i in range(n_all):
    invs_all.append(inv_ttdata_all[i].flatten().tolist())
    refs_all.append(ref_ttdata_all[i].flatten().tolist())
invs_all = np.array(invs_all).flatten()
refs_all = np.array(refs_all).flatten()

max_t = max(np.max(invs_all), np.max(refs_all))
x_min_val = 0
x_max_val = max_t + 0.25 * max_t

data_sequences = [
    {"name": "L1", "ref_data": ref_data["l0"], "inv_data": inv_data["l0"],
     "color": TABLEAU_COLORS[0], "marker": MARKERS["L0"][0], "level": "Lv. 1", "fig_indx": "a"},
    {"name": "L2", "ref_data": ref_data["l1"], "inv_data": inv_data["l1"],
     "color": TABLEAU_COLORS[3], "marker": MARKERS["L1"][0], "level": "Lv. 2", "fig_indx": "b"},
    {"name": "L3", "ref_data": ref_data["l2"], "inv_data": inv_data["l2"],
     "color": TABLEAU_COLORS[7], "marker": MARKERS["L2"][0], "level": "Lv. 3", "fig_indx": "c"},
    {"name": "L4", "ref_data": ref_data["l3"], "inv_data": inv_data["l3"],
     "color": TABLEAU_COLORS[10], "marker": MARKERS["L3"][0], "level": "Lv. 4", "fig_indx": "d"},
    {"name": "L5", "ref_data": ref_data["l4"], "inv_data": inv_data["l4"],
     "color": TABLEAU_COLORS[12], "marker": MARKERS["L4"][0], "level": "Lv. 5", "fig_indx": "e"},
]

output_dir = f"./figure_table_output/Figure/Figure5/scenario_{scenario}_case{case_num}"
os.makedirs(output_dir, exist_ok=True)

generate_all_consistent_plots(
    data_sequences, x_min=x_min_val, x_max=x_max_val,
    savepath=output_dir, fig_indx_show=False,
)
print(f"[5b-g] Level-based figures saved to {output_dir}")

legend_path = os.path.join(os.path.dirname(output_dir), "plot_legend")
create_tt_level_legend(data_sequences, savepath=legend_path)
print(f"[5b-g] Legend saved to {legend_path}")

print("\nDone! All figures generated.")
