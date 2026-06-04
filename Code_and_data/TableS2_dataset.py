import os
import numpy as np
import pandas as pd
from dataFunc.statisticalProcess import read_refdata, calculate_statiscal_indicators
from geostatFunc.statistics import calculate_statistics, format_numeric_columns

# ============================================================
#  Configuration
# ============================================================
field_1215 = "dataset_scenarios"
field_0111 = "dataset_scenarios"

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

# 10 hetero-field cases (same as original boxplot scenario_case)
hetero_cases = [
    {"field": field_1215, "scenario": scenarios[0], "case": cases_name[0]},
    {"field": field_1215, "scenario": scenarios[0], "case": cases_name[1]},
    {"field": field_1215, "scenario": scenarios[0], "case": cases_name[1]},
    {"field": field_1215, "scenario": scenarios[1], "case": cases_name[2]},
    {"field": field_1215, "scenario": scenarios[1], "case": cases_name[3]},
    {"field": field_1215, "scenario": scenarios[1], "case": cases_name[4]},
    {"field": field_1215, "scenario": scenarios[2], "case": cases_name[5]},
    {"field": field_1215, "scenario": scenarios[0], "case": cases_name[1]},
    {"field": field_1215, "scenario": scenarios[2], "case": cases_name[6]},
    {"field": field_1215, "scenario": scenarios[2], "case": cases_name[7]},
]

# 2 mean-field (1a) cases using corrected invsResult data
meanfield_scenario = "1a"
meanfield_cases = [
    {"field": field_0111, "scenario": meanfield_scenario, "case": "mean_-11.5"},
    {"field": field_0111, "scenario": meanfield_scenario, "case": "mean_-9.2"},
]

all_cases = hetero_cases + meanfield_cases

output_dir = "./figure_table_output/Table"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "TableS2_stats_dataset.xlsx")

# ============================================================
#  Compute and export
# ============================================================
csi = calculate_statiscal_indicators

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    for entry in all_cases:
        field_name = entry["field"]
        scenario = entry["scenario"]
        case_name = entry["case"]

        if scenario == "1a":
            invs_dir = "invsResult"
            ref_dir = "homo_field"
        else:
            invs_dir = "invsResult"
            ref_dir = "hetero_field"

        inv_path = f"./data/{field_name}/{scenario}/{case_name}/lnKs/{invs_dir}"
        ref_path = f"./data/{field_name}/{scenario}/{case_name}/lnKs/{ref_dir}"

        invdata = read_refdata(inv_path)
        refdata = read_refdata(ref_path)

        inv_Ks = np.array(invdata.data)
        ref_Ks = np.array(refdata.data)

        n_real = len(inv_Ks)

        rmse_list, mbe_list = [], []
        ssim_g_list, ssim_l_list = [], []
        slope_list, intercept_list = [], []
        r_value_list, r2_value_list = [], []
        p_value_list, std_err_list = [], []

        for n in range(n_real):
            try:
                (
                    rmse,
                    mbe,
                    ssim_g,
                    ssim_l,
                    ssim_l_map,
                    slope,
                    intercept,
                    r_value,
                    r2_value,
                    p_value,
                    std_err,
                ) = csi(ref_Ks[n], inv_Ks[n])
            except ValueError:
                rmse = np.sqrt(np.mean((inv_Ks[n] - ref_Ks[n]) ** 2))
                mbe = np.mean(inv_Ks[n] - ref_Ks[n])
                ssim_g = ssim_l = ssim_l_map = np.nan
                slope = intercept = r_value = r2_value = p_value = std_err = np.nan

            rmse_list.append(rmse)
            mbe_list.append(mbe)
            ssim_g_list.append(ssim_g)
            ssim_l_list.append(ssim_l)
            slope_list.append(slope)
            intercept_list.append(intercept)
            r_value_list.append(r_value)
            r2_value_list.append(r2_value)
            p_value_list.append(p_value)
            std_err_list.append(std_err)

        stats_indicators_dict = {
            "Realization": list(range(0, n_real)),
            "RMSE": rmse_list,
            "MBE": mbe_list,
            "Pearson_r": r_value_list,
            "Global_SSIM": ssim_g_list,
            "Local_SSIM": ssim_l_list,
            "R_squared": r2_value_list,
            "Slope": slope_list,
            "Intercept": intercept_list,
            "p_value": p_value_list,
            "Std_err": std_err_list,
        }
        stats_indicators_df = pd.DataFrame(stats_indicators_dict)

        # Write raw indicators sheet
        sheet_name = f"{scenario}_{case_name}"
        stats_indicators_df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"  sheet: {sheet_name} ({n_real} realizations)")

        # Compute and write formatted statistics sheet
        statistics_dict = calculate_statistics(stats_indicators_df)
        statistics_df = pd.DataFrame(statistics_dict).T
        statistics_df = statistics_df.reset_index().rename(
            columns={"index": "Parameter"}
        )
        statistics_df["scenario"] = scenario
        statistics_df["case"] = case_name

        column_order = [
            "scenario",
            "case",
            "Parameter",
            "mean",
            "min",
            "max",
            "std",
            "median",
            "count",
            "missing",
        ]
        available_columns = [
            col for col in column_order if col in statistics_df.columns
        ]
        statistics_df = statistics_df[available_columns]
        statistics_df_formatted = format_numeric_columns(
            statistics_df, decimal_places=2
        )

        statistics_df_formatted["range"] = statistics_df.apply(
            lambda row: (
                f"[{row['min']:.2f}, {row['max']:.2f}]"
                if pd.notnull(row["min"]) and pd.notnull(row["max"])
                else ""
            ),
            axis=1,
        )

        column_order_formatted = [
            "scenario",
            "case",
            "Parameter",
            "mean",
            "range",
            "std",
            "min",
            "max",
            "median",
            "count",
            "missing",
        ]
        available_columns_formatted = [
            col
            for col in column_order_formatted
            if col in statistics_df_formatted.columns
        ]
        statistics_df_formatted = statistics_df_formatted[available_columns_formatted]

        stats_sheet_name = f"{scenario}_{case_name}_stats"
        statistics_df_formatted.to_excel(
            writer, sheet_name=stats_sheet_name, index=False
        )
        print(f"  sheet: {stats_sheet_name}")

print(f"\nAll data exported to {output_file}")
print("finish!")
