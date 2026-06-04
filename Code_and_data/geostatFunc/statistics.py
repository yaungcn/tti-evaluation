import numpy as np
import pandas as pd
from dataFunc.statisticalProcess import read_refdata, calculate_statiscal_indicators


def calculate_statistics(df):
    """
    Compute per-column statistics for a DataFrame.
    """
    statistics = {}
    for column in df.columns:
        data = df[column].dropna()
        if len(data) > 0:
            stats = {
                "mean": np.mean(data),
                "min": np.min(data),
                "max": np.max(data),
                "std": np.std(data, ddof=1),
                "median": np.median(data),
                "count": len(data),
                "missing": df[column].isna().sum(),
            }
        else:
            stats = {
                "mean": np.nan,
                "min": np.nan,
                "max": np.nan,
                "std": np.nan,
                "median": np.nan,
                "count": 0,
                "missing": len(df[column]),
            }
        statistics[column] = stats
    return statistics


def format_numeric_columns(df, decimal_places=2):
    """
    Format numeric columns to specified decimal places.
    """
    df_formatted = df.copy()
    for column in df_formatted.columns:
        if pd.api.types.is_numeric_dtype(df_formatted[column]):
            df_formatted[column] = df_formatted[column].apply(
                lambda x: f"{x:.{decimal_places}f}" if pd.notnull(x) else ""
            )
    return df_formatted


def setup_case_2a(case_data, data_label, scenario_case, field_name):
    """
    Load and compute metric lists for scenario 2a (2 cases).
    Populates case_data in-place.
    """
    csi = calculate_statiscal_indicators
    case_ind = 0
    for sce_case in scenario_case:
        scenario = sce_case["scenario"]
        case_name = sce_case["case"]

        invdata_lnKs = read_refdata(
            f"./data/{field_name}/{scenario}/{case_name}/lnKs/invsResult"
        )
        refdata_lnKs = read_refdata(
            f"./data/{field_name}/{scenario}/{case_name}/lnKs/hetero_field"
        )

        ref_Ks = np.array(refdata_lnKs.data)
        inv_Ks = np.array(invdata_lnKs.data)

        rmse_list, mbe_list, r_value_list, slope_list = [], [], [], []

        for n in range(200):
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

            rmse_list.append(rmse)
            mbe_list.append(mbe)
            r_value_list.append(r_value)
            slope_list.append(slope)

        case_data[case_ind][data_label[0]] = rmse_list
        case_data[case_ind][data_label[1]] = mbe_list
        case_data[case_ind][data_label[2]] = r_value_list
        case_data[case_ind][data_label[3]] = slope_list

        print(f"scenario {scenario} case {case_ind}")
        case_ind += 1


def setup_case_2bc(case_data, data_label, scenario_case, field_name):
    """
    Load and compute metric lists for scenarios 2b/2c (4 cases each).
    Populates case_data in-place.
    """
    csi = calculate_statiscal_indicators
    case_ind = 0
    for sce_case in scenario_case:
        scenario = sce_case["scenario"]
        case_name = sce_case["case"]

        invdata_lnKs = read_refdata(
            f"./data/{field_name}/{scenario}/{case_name}/lnKs/invsResult"
        )
        refdata_lnKs = read_refdata(
            f"./data/{field_name}/{scenario}/{case_name}/lnKs/hetero_field"
        )

        ref_Ks = np.array(refdata_lnKs.data)
        inv_Ks = np.array(invdata_lnKs.data)

        rmse_list, mbe_list, r_value_list, slope_list = [], [], [], []

        for n in range(200):
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

            rmse_list.append(rmse)
            mbe_list.append(mbe)
            r_value_list.append(r_value)
            slope_list.append(slope)

        case_data[case_ind][data_label[0]] = rmse_list
        case_data[case_ind][data_label[1]] = mbe_list
        case_data[case_ind][data_label[2]] = r_value_list
        case_data[case_ind][data_label[3]] = slope_list

        print(f"scenario {scenario} case {case_ind}")
        case_ind += 1
