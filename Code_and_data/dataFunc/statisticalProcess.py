import numpy as np
import pandas as pd
import os

# import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error  # r2_score
from skimage.metrics import structural_similarity as ssim
from scipy import stats


def read_ttdata(folder):
    """# 读取反演结果文件夹下所有文件夹内走时文件ttdata.csv
    # 以文件名为各数据名称，DataFrame"""
    data_list = []
    name_list = []
    print(f"read ttdata.csv from {folder}")
    for root, dirs, files in os.walk(folder):
        # print("root", root)
        # print(f"read ttdata.csv from {root}")
        for filename in files:
            if filename == "ttdata.csv":
                file_path = os.path.join(root, filename)
                # print("file path", file_path)
                file_data = np.loadtxt(file_path, delimiter=",")
                name_list.append(os.path.basename(root))
                data_list.append(file_data)

    ttdata = pd.DataFrame({"name": name_list, "data": data_list})
    return ttdata


def read_correct_ttdata(folder):
    """# 读取反演结果文件夹下所有文件夹内走时文件ttdata.csv，并对对走时进行修正
    # 返回 DataFrame：
    # ttdata = pd.DataFrame(
        {
            "name": name_list,
            "data": data_list,
            "all_data": all_data_list,
            "correct_data": correct_data_list,
        })"""
    data_list = []
    name_list = []
    all_data_list = []
    all_name_list = []
    correct_data_list = []
    print(f"read corrected ttdata.csv from {folder}")
    for root, dirs, files in os.walk(folder):
        for dir in dirs:
            filename_obs = "ttdata.csv"
            file_path = os.path.join(root, dir, filename_obs)
            # print("file_path", file_path)
            file_data = np.loadtxt(file_path, delimiter=",")
            name_list.append(os.path.basename(root))
            data_list.append(file_data)

            filename_all = "all_ttdata.csv"
            all_file_path = os.path.join(root, dir, filename_all)
            # print("all_file_path", all_file_path)
            all_file_data = np.loadtxt(all_file_path, delimiter=",")
            all_name_list.append(os.path.basename(folder))
            all_data_list.append(all_file_data)

            correct_data = file_data - np.min(all_file_data, axis=1, keepdims=True)
            correct_data_list.append(correct_data)

    ttdata = pd.DataFrame(
        {
            "name": name_list,
            "data": data_list,
            "all_data": all_data_list,
            "correct_data": correct_data_list,
        }
    )
    return ttdata


# def read_ttdata_homo(folder):
#     """# 读取反演结果文件夹下所有文件夹内走时文件ttdata.csv
#     # 以文件名为各数据名称，DataFrame"""
#     data_list = []
#     name_list = []
#     print(f"read ttdata.csv from {folder}")
#     for root, dirs, files in os.walk(folder):
#         # print(f"read ttdata.csv from {root}")
#         for filename in files:
#             if filename == "ttdata.csv":
#                 file_path = os.path.join(root, filename)
#                 file_data = np.loadtxt(file_path, delimiter=",")
#                 name_list.append(os.path.basename(root))
#                 data_list.append(file_data)

#     ttdata = pd.DataFrame({"name": name_list, "data": data_list})
#     return ttdata


def read_invdata(folder):
    """# 读取反演结果文件夹下所有文件夹内反演结果文件inv_HTT_result.txt
    # 以文件名为各数据名称，DataFrame"""
    names = []
    data_list = []

    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename == "inv_resultD.txt":
                file_path = os.path.join(root, filename)
                try:
                    # 根据实际文件格式调整参数，例如 delimiter, skiprows 等
                    file_data = np.loadtxt(file_path)
                    names.append(os.path.basename(root))
                    data_list.append((file_data))
                except Exception as e:
                    print(f"警告：读取文件 {file_path} 时出错: {e}")
                    # 跳过这个文件，继续处理下一个

    # 返回一个结构更清晰的字典
    invdata = pd.DataFrame({"name": names, "data": data_list})
    return invdata


def read_log_invdata(folder):
    """# 读取反演结果文件夹下所有文件夹内反演结果文件inv_HTT_result.txt的log值
    # 以文件名为各数据名称，DataFrame"""
    names = []
    data_list = []

    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename == "inv_resultD.txt":
                file_path = os.path.join(root, filename)
                try:
                    # 根据实际文件格式调整参数，例如 delimiter, skiprows 等
                    file_data = np.loadtxt(file_path)
                    names.append(os.path.basename(root))
                    data_list.append(np.log(file_data))
                except Exception as e:
                    print(f"警告：读取文件 {file_path} 时出错: {e}")
                    # 跳过这个文件，继续处理下一个

    # 返回一个结构更清晰的字典
    invdata = pd.DataFrame({"name": names, "data": data_list})
    return invdata


def read_refdata(folder):
    """
    从指定文件夹读取所有txt文件，并将数据存储为DataFrame

    参数:
    folder: 要遍历的文件夹路径

    返回:
    pd.DataFrame: 包含'name'和'data'两列的DataFrame
    """
    # 创建列表来存储数据
    names = []
    data_list = []

    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                try:
                    # 读取文件，跳过前4行
                    file_data = np.loadtxt(file_path, skiprows=4)
                    # 使用文件名（不含扩展名）作为name
                    key = os.path.splitext(filename)[0]

                    # 添加到列表
                    names.append(key)
                    data_list.append(file_data)

                except Exception as e:
                    print(f"警告：读取文件 {file_path} 时出错: {e}")
                    continue

    # 创建DataFrame
    ref_data = pd.DataFrame({"name": names, "data": data_list})

    return ref_data


def calculate_ssim_skimage(
    img1, img2, normalize=False, local_normalize=False, winsize=None
):
    """
    使用scikit-image计算SSIM

    参数:
    img1, img2: 输入图像（二维或三维数组）
    normalize: 是否全局归一化到0-1范围
    local_normalize: 是否局部归一化

    返回:
    ssim_value: SSIM值 (-1到1之间，1表示完全相同)
    """

    if normalize:
        # 归一化到0-1范围
        img1 = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-10)
        img2 = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-10)

    if not local_normalize:
        # 计算SSIM
        combined_min = min(img1.min(), img2.min())
        combined_max = max(img1.max(), img2.max())
        data_range_val = combined_max - combined_min
        ssim_value = ssim(
            img1,
            img2,
            data_range=data_range_val,  # img1.max() - img1.min(),
            win_size=winsize,  # 自动计算
            channel_axis=-1 if img1.ndim == 3 else None,
        )
        return ssim_value

    else:
        combined_min = min(img1.min(), img2.min())
        combined_max = max(img1.max(), img2.max())
        data_range_val = combined_max - combined_min

        print(f"图像1范围: [{img1.min():.4f}, {img1.max():.4f}]")
        print(f"图像2范围: [{img2.min():.4f}, {img2.max():.4f}]")
        print(f"使用的 data_range: {data_range_val:.4f}")
        mssim, ssim_map = ssim(
            img1,
            img2,
            data_range=data_range_val,  # 使用局部归一化
            win_size=winsize,
            gaussian_weights=True,
            full=True,
        )
        return mssim, ssim_map


def calculate_statiscal_indicators(ref, inv, ssim_win_size=5):
    """
    计算参考场与估计场之间的量化指标

    参数：
        ref：参考场
        inv：估计场

    返回：
        rmse,
        mbe,
        ssim_global,
        ssim_local,
        ssim_local_map,
        slope,
        intercept,
        r_value,
        r2_value,
        p_value,
        std_err
    """
    ref_list = ref.flatten().tolist()
    inv_list = inv.flatten().tolist()
    # rmse, mbe
    rmse = np.sqrt(mean_squared_error(ref_list, inv_list))
    mbe = np.mean(inv - ref)

    # r, ssim
    # corr_matrix = np.corrcoef(ref_list, inv_list)
    # r = corr_matrix[0, 1]

    combined_min = min(ref.min(), inv.min())
    combined_max = max(ref.max(), inv.max())
    data_range_val = combined_max - combined_min

    ssim_global = ssim(
        ref, inv, data_range=data_range_val, win_size=None  # img1.max() - img1.min(),
    )
    ssim_local, ssim_local_map = ssim(
        ref,
        inv,
        data_range=data_range_val,  # img1.max() - img1.min(),
        win_size=ssim_win_size,  # 自动计算的3 - ssim_ln_map
        gaussian_weights=True,
        full=True,
    )

    # fit, r2

    # fit = np.polyfit(ref_list, inv_list, 1)
    # pred = np.poly1d(fit)
    # inv_fit = pred(ref_list)
    # r2 = r2_score(inv_list, inv_fit)
    # slope = fit[0]
    # intercept = fit[1]

    slope, intercept, r_value, p_value, std_err = stats.linregress(ref_list, inv_list)
    r2_value = r_value**2

    return (
        rmse,
        mbe,
        ssim_global,
        ssim_local,
        ssim_local_map,
        slope,
        intercept,
        r_value,
        r2_value,
        p_value,
        std_err,
    )
