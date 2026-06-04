import os
import csv
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import glob
import toml
from pathlib import Path


def read_folders(path):
    """Read all folder names in the given path, return a list of folder names."""
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


def process_export_files(work_path):
    """Process export th=*_traveltime_data.csv files into ttdata.csv file in the given work_path directory structure."""
    field_case_folder_names = read_folders(work_path)
    for i, folder in enumerate(field_case_folder_names):
        case_folder_names = read_folders(os.path.join(work_path, folder))
        print(folder)
        for j, case in enumerate(case_folder_names):
            case_export_path = os.path.join(work_path, folder, case, "model_export")
            print(case)
            seed_path = read_folders(case_export_path)
            # print(seed_path)
            for k, seed in enumerate(seed_path):
                seed_case_path = os.path.join(case_export_path, seed)
                print(f"处理文件夹 {k+1}/{len(seed_path)}: {seed}")

                # 检查是否为目录
                if not os.path.isdir(seed_case_path):
                    print(f"跳过非目录: {seed_case_path}")
                    continue

                files = [
                    f
                    for f in os.listdir(seed_case_path)
                    if os.path.isfile(os.path.join(seed_case_path, f))
                    and f.endswith("traveltime_data.csv")
                ]

                print(f"找到traveltime_data.csv文件: {files}")

                if not files:
                    print(f"文件夹 {seed_case_path} 中没有找到traveltime_data.csv文件")
                    continue

                    # 创建或清空目标CSV文件

                datafile = os.path.join(seed_case_path, "dhdt_th_ttdata.csv")

                with open(datafile, "w", encoding="utf-8") as csv_file:
                    csv_file.write("")  # 清空文件

                processed_files = 0

                # 读取每个文件夹下的CSV文件内容
                for file in files:
                    if file in ["ttdata.csv"]:  # 跳过输出文件
                        continue
                    file_path = os.path.join(seed_case_path, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            lines = content.splitlines()
                            if len(lines) > 5:
                                sixth_line = lines[5]
                                # 将第六行保存到CSV中
                                with open(datafile, "a", encoding="utf-8") as csv_file:
                                    csv_file.write(f"{sixth_line}\n")
                                processed_files += 1
                                print(f"  处理文件: {file} ✓")
                            else:
                                print(f"  文件 {file} 行数不足6行")

                    except Exception as e:
                        print(f"  读取文件 {file} 时出错: {e}")

                print(f"处理了 {processed_files} 个文件")

                # 处理完所有文件后，读取CSV并提取指定列

                try:
                    if os.path.exists(datafile) and processed_files > 0:
                        data = np.loadtxt(datafile, delimiter=",")
                        print(f"  成功读取数据，形状: {data.shape}")
                        # 确保数据至少有57列
                        columns = 57
                        if data.ndim == 1:  # 如果只有一行数据
                            data = data.reshape(1, -1)
                        if data.shape[1] >= columns:
                            # 提取所有走时
                            all_tt = data[:, 2:57:3]

                            # 提取前5列走时和后5列走时（索引2, 5, 8, 11, 14, 44, 47, 50, 53, 56）
                            selected_columns_tt = data[
                                :, [2, 5, 8, 11, 14, 44, 47, 50, 53, 56]
                            ]
                            # 提取前5列水头和后5列水头（索引0, 3, 6, 9, 12, 42, 45, 48, 51, 54）
                            selected_columns_hd = data[
                                :, [0, 3, 6, 9, 12, 42, 45, 48, 51, 54]
                            ]

                            output_file_all_tt = os.path.join(
                                seed_case_path, "all_ttdata.csv"
                            )
                            output_file_tt = os.path.join(seed_case_path, "ttdata.csv")
                            output_file_hd = os.path.join(seed_case_path, "hddata.csv")

                            np.savetxt(
                                output_file_all_tt,
                                all_tt,
                                delimiter=",",
                                fmt="%.6f",
                            )
                            np.savetxt(
                                output_file_tt,
                                selected_columns_tt,
                                delimiter=",",
                                fmt="%.6f",
                            )
                            np.savetxt(
                                output_file_hd,
                                selected_columns_hd,
                                delimiter=",",
                                fmt="%.6f",
                            )
                            print(
                                f"  ✓ 成功保存提取的数据到 all_ttdata.csv，形状: {all_tt.shape}"
                            )
                            print(
                                f"  ✓ 成功保存提取的数据到 ttdata.csv，形状: {selected_columns_tt.shape}"
                            )
                            print(
                                f"  ✓ 成功保存提取的数据到 hddata.csv，形状: {selected_columns_hd.shape}"
                            )
                        else:
                            print(
                                f"  ✗ 数据列数不足{columns}列，实际为 {data.shape[1]} 列"
                            )
                    else:
                        print(f"  没有有效数据可处理")
                except Exception as e:
                    print(f"  处理CSV文件时出错: {e}")


def check_value_in_csv(filename, target=60):
    """check if the target value exists in the CSV file, return True or False"""
    with open(filename, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            for item in row:
                try:
                    value = float(item)
                    if value == target:
                        return True
                except ValueError:
                    continue
    return False


# statistical function of the traveltime data
def read_all_ttdata(folder):
    """read all ttdata.csv travel-time data files in the folder and subfolders"""
    data = {}
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename == "ttdata.csv":
                file_path = os.path.join(root, filename)
                file_data = np.loadtxt(file_path, delimiter=",")
                data[os.path.basename(root)] = file_data
    return data


def read_all_ref(folder):
    """read all .txt ref files in the folder and subfolders, return a dictionary with filename (without extension) as key and data as value"""
    data = {}
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                # print(f"Reading file: {file_path}")
                file_data = np.loadtxt(file_path, skiprows=4)
                # erase the file extension and use the filename as the key
                key = os.path.splitext(filename)[0]
                data[key] = file_data
    return data


def read_ref(folder, num=1):
    """read a single .txt ref file with <num>, return the data"""
    # 检索有no_{num}前缀的文件
    folder_files = os.listdir(folder)
    # 设置文件名前缀为四位数，不足补零
    num_str = str(num).zfill(4)
    file_names = [
        f for f in folder_files if f.startswith(f"no_{num_str}") and f.endswith(".txt")
    ]
    if not file_names:
        raise FileNotFoundError(f"No file with no_{num}.txt found in {folder}")
    file_path = os.path.join(folder, file_names[0])
    file_data = np.loadtxt(file_path, skiprows=4)
    file_name = file_names[0] if file_names else ""
    return file_data, file_name


def read_all_est(folder):
    """read all inv_HTT_result.txt files in the folder and subfolders, return a dictionary with folder name as key and data as value"""
    data = {}
    for root, dirs, files in os.walk(folder):
        for dir in dirs:
            # print(dir)
            file_path = os.path.join(root, dir, "inv_HTT_result.txt")

            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            print(f"Reading file: {file_path}")
            file_data = read_est(root, dir)
            # print(file_data)
            # erase the file extension and use the filename as the key
            data[dir] = file_data
    return data


def read_est(folder_case, case_seed, invs_file_name="inv_HTT_result.txt"):
    """read a single inv_HTT_result.txt file in the folder/case, return the data"""
    file_path = os.path.join(folder_case, case_seed, invs_file_name)
    file_data = np.loadtxt(file_path)
    return file_data


def mean_all_data(data):
    """calculate the mean of all data in the dictionary, return a new dictionary with the same keys and mean values"""
    mean_data = np.zeros((10, 10))
    for key, value in data.items():
        mean_data += value
    mean_data = mean_data / len(data)
    return mean_data


def cal_R2_L1_L2(ref_data, est_data):
    """calculate the R2, L1, L2 between the reference data and estimated data"""
    if ref_data.shape != est_data.shape:
        raise ValueError(
            "The shape of reference data and estimated data must be the same"
        )
    # 计算R²
    ss_res = np.sum((ref_data - est_data) ** 2)
    ss_tot = np.sum((ref_data - np.mean(ref_data)) ** 2)

    r2 = r2_score(ref_data.flatten(), est_data.flatten())

    # 计算L1范数
    # l1 = np.sum(np.abs(ref_data - est_data))/ref_data.size
    mae = mean_absolute_error(ref_data.flatten(), est_data.flatten())

    # 计算L2范数
    # l2 = np.sqrt(np.sum((ref_data - est_data) ** 2)) / ref_data.size
    mse = mean_squared_error(ref_data.flatten(), est_data.flatten())  # MSE
    rmse = np.sqrt(mse)  # RMSE

    return r2, mae, mse, rmse


def load_toml_files_glob(work_path):
    """使用glob加载所有匹配的TOML文件"""
    # 构建通配符路径
    pattern = os.path.join(work_path, "*.toml")

    # 查找所有匹配的文件
    toml_files = glob.glob(pattern)

    if not toml_files:
        raise FileNotFoundError(f"没有找到TOML文件: {pattern}")

    configs = {}
    for file_path in toml_files:
        try:
            config = toml.load(file_path)
            file_name = os.path.basename(file_path)
            configs = config
            print(f"已加载: {file_name}")
        except Exception as e:
            print(f"加载失败 {file_path}: {e}")

    return configs


def process_invs_files(work_path):
    """Process inverse result inv_resultD.txt files into no_0000_seed_00000.txt file in the given work_path directory structure."""
    field_case_folder_names = read_folders(work_path)
    for i, folder in enumerate(field_case_folder_names):
        case_folder_names = read_folders(os.path.join(work_path, folder))
        print(folder)
        for j, case in enumerate(case_folder_names):
            case_export_path = os.path.join(work_path, folder, case, "model_export")
            print(case)
            seed_path = read_folders(case_export_path)
            # print(seed_path)
            field_config_path = os.path.join(work_path, folder, case)
            field_config = load_toml_files_glob(field_config_path)
            # print(field_config["field_params_Gaussian_case_1a.toml"])
            mean = field_config["fields"][1]["mean"]
            # print(mean)
            size = field_config["grid"]["size"]
            grid_num = field_config["grid"]["num"]

            min_cellcenter = -(size / 2 - size / grid_num / 2)
            max_cellcenter = size / 2 - size / grid_num / 2

            if size <= 0:
                raise ValueError("Size must be a positive integer.")
            if min_cellcenter >= max_cellcenter:
                raise ValueError("min_cellcenter must be less than max_cellcenter.")
            grid_center_x_limit = [min_cellcenter, max_cellcenter]
            grid_center_y_limit = [min_cellcenter, max_cellcenter]
            grid_center_x = np.linspace(
                grid_center_x_limit[0], grid_center_x_limit[1], grid_num
            )
            grid_center_y = np.linspace(
                grid_center_y_limit[0], grid_center_y_limit[1], grid_num
            )
            for k, seed in enumerate(seed_path):
                seed_case_path = os.path.join(case_export_path, seed)
                print(f"处理文件夹 {k+1}/{len(seed_path)}: {seed}")

                # 检查是否为目录
                if not os.path.isdir(seed_case_path):
                    print(f"跳过非目录: {seed_case_path}")
                    continue

                inv_export_path = os.path.join(
                    work_path, folder, case, "lnKs", "invsResult"
                )
                result_file = os.path.join(seed_case_path, "inv_resultD.txt")

                if not os.path.exists(result_file):
                    print(f"文件夹 {seed_case_path} 中没有找到inv_resultD.txt文件")
                    continue

                est_D = np.loadtxt(result_file)
                log_est_D = np.log(np.array(est_D))
                log_est_K = log_est_D + mean

                if not os.path.isdir(inv_export_path):
                    try:
                        # 使用makedirs替代mkdir
                        os.makedirs(inv_export_path, exist_ok=True)
                        print(f"已创建目录: {inv_export_path}")
                    except Exception as e:
                        print(f"创建目录失败: {e}")
                        # 根据需求决定是否继续
                        # continue  # 如果创建失败，可能需要跳过后续处理
                        return  # 或者直接返回

                datafile = os.path.join(inv_export_path, f"{seed}.txt")

                with open(datafile, "w", encoding="utf-8") as csv_file:
                    csv_file.write("")  # 清空文件

                with open(datafile, "w", encoding="utf-8") as f:
                    f.write("%(x,y)-grid\n")
                    np.savetxt(f, np.array([grid_center_x, grid_center_y]), fmt="%.6f")
                    f.write("% data\n")
                    np.savetxt(f, log_est_K, fmt="%.6f")
    print("---Finish process inverse result---")


def process_invs_export_files(work_path):
    """Process export th=*_traveltime_data.csv files into ttdata.csv file in the given work_path directory structure."""
    field_case_folder_names = read_folders(work_path)
    for i, folder in enumerate(field_case_folder_names):
        case_folder_names = read_folders(os.path.join(work_path, folder))
        print(folder)
        for j, case in enumerate(case_folder_names):
            case_export_path = os.path.join(work_path, folder, case, "model_export_inv")
            print(case)
            seed_path = read_folders(case_export_path)
            # print(seed_path)
            for k, seed in enumerate(seed_path):
                seed_case_path = os.path.join(case_export_path, seed)
                print(f"处理文件夹 {k+1}/{len(seed_path)}: {seed}")

                # 检查是否为目录
                if not os.path.isdir(seed_case_path):
                    print(f"跳过非目录: {seed_case_path}")
                    continue

                files = [
                    f
                    for f in os.listdir(seed_case_path)
                    if os.path.isfile(os.path.join(seed_case_path, f))
                    and f.endswith("traveltime_data.csv")
                ]

                print(f"找到traveltime_data.csv文件: {files}")

                if not files:
                    print(f"文件夹 {seed_case_path} 中没有找到traveltime_data.csv文件")
                    continue

                    # 创建或清空目标CSV文件

                datafile = os.path.join(seed_case_path, "dhdt_th_ttdata.csv")

                with open(datafile, "w", encoding="utf-8") as csv_file:
                    csv_file.write("")  # 清空文件

                processed_files = 0

                # 读取每个文件夹下的CSV文件内容
                for file in files:
                    if file in ["ttdata.csv"]:  # 跳过输出文件
                        continue
                    file_path = os.path.join(seed_case_path, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            lines = content.splitlines()
                            if len(lines) > 5:
                                sixth_line = lines[5]
                                # 将第六行保存到CSV中
                                with open(datafile, "a", encoding="utf-8") as csv_file:
                                    csv_file.write(f"{sixth_line}\n")
                                processed_files += 1
                                print(f"  处理文件: {file} ✓")
                            else:
                                print(f"  文件 {file} 行数不足6行")

                    except Exception as e:
                        print(f"  读取文件 {file} 时出错: {e}")

                print(f"处理了 {processed_files} 个文件")

                # 处理完所有文件后，读取CSV并提取指定列

                try:
                    if os.path.exists(datafile) and processed_files > 0:
                        data = np.loadtxt(datafile, delimiter=",")
                        print(f"  成功读取数据，形状: {data.shape}")
                        # 确保数据至少有57列
                        columns = 57
                        if data.ndim == 1:  # 如果只有一行数据
                            data = data.reshape(1, -1)
                        if data.shape[1] >= columns:
                            # 提取所有走时
                            all_tt = data[:, 2:57:3]

                            # 提取前5列走时和后5列走时（索引2, 5, 8, 11, 14, 44, 47, 50, 53, 56）
                            selected_columns_tt = data[
                                :, [2, 5, 8, 11, 14, 44, 47, 50, 53, 56]
                            ]
                            # 提取前5列水头和后5列水头（索引0, 3, 6, 9, 12, 42, 45, 48, 51, 54）
                            selected_columns_hd = data[
                                :, [0, 3, 6, 9, 12, 42, 45, 48, 51, 54]
                            ]

                            output_file_all_tt = os.path.join(
                                seed_case_path, "all_ttdata.csv"
                            )
                            output_file_tt = os.path.join(seed_case_path, "ttdata.csv")
                            output_file_hd = os.path.join(seed_case_path, "hddata.csv")

                            np.savetxt(
                                output_file_all_tt,
                                all_tt,
                                delimiter=",",
                                fmt="%.6f",
                            )
                            np.savetxt(
                                output_file_tt,
                                selected_columns_tt,
                                delimiter=",",
                                fmt="%.6f",
                            )
                            np.savetxt(
                                output_file_hd,
                                selected_columns_hd,
                                delimiter=",",
                                fmt="%.6f",
                            )
                            print(
                                f"  ✓ 成功保存提取的数据到 all_ttdata.csv，形状: {all_tt.shape}"
                            )
                            print(
                                f"  ✓ 成功保存提取的数据到 ttdata.csv，形状: {selected_columns_tt.shape}"
                            )
                            print(
                                f"  ✓ 成功保存提取的数据到 hddata.csv，形状: {selected_columns_hd.shape}"
                            )
                        else:
                            print(
                                f"  ✗ 数据列数不足{columns}列，实际为 {data.shape[1]} 列"
                            )
                    else:
                        print(f"  没有有效数据可处理")
                except Exception as e:
                    print(f"  处理CSV文件时出错: {e}")


if __name__ == "__main__":
   pass
