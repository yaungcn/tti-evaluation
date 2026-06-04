import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error


def draw_est_ref_tt_comparison(
    scenario_case,
    ref_D,
    inv_D,
    param_name="travel time",
    unit="s",
    num_all_case=200,
    save=False,
    save_path="./",
    show_fig_indx=False,
    text_xy=[0.05, 0.95],
):

    # 创建图形和坐标轴
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["mathtext.fontset"] = "stixsans"  # STIX字体与Times兼容
    plt.rcParams.update(
        {
            "font.size": 20,  # 全局字体大小
            "axes.titlesize": 20,  # 坐标轴标题大小
            "axes.labelsize": 20,  # 坐标轴标签大小
            "xtick.labelsize": 20,  # x轴刻度标签大小
            "ytick.labelsize": 20,  # y轴刻度标签大小
            "legend.fontsize": 16,  # 图例字体大小
        }
    )
    plt.rcParams["axes.linewidth"] = 2.0

    fig, ax0 = plt.subplots(figsize=(10, 8))

    n_all = num_all_case
    invs_all = []
    refs_all = []

    for i in range(n_all):
        invs_all.append((inv_D[i].flatten().tolist()))
        refs_all.append((ref_D[i].flatten().tolist()))

    invs_all = np.array(invs_all).flatten().tolist()
    invs_all = np.array(invs_all)
    refs_all = np.array(refs_all).flatten().tolist()
    refs_all = np.array(refs_all)

    max_D_all = max(np.max(invs_all), np.max(refs_all))
    # min_D_all = min(np.min(invs_all), np.min(refs_all))

    xlim_all = [0, int(max_D_all) * 1.1 + 1.5]
    ylim_all = [0, int(max_D_all) * 1.1 + 1.5]

    max_min = abs(xlim_all[0] - xlim_all[1])
    if 25 > max_min and max_min > 10:
        axis_scale = 2
    elif max_min > 40:
        axis_scale = 5
    else:
        axis_scale = 1

    ax0.set_xticks(np.arange(xlim_all[0], xlim_all[1], axis_scale))
    ax0.set_yticks(np.arange(ylim_all[0], ylim_all[1], axis_scale))

    ax0.scatter(
        refs_all,
        invs_all,
        label="Est. vs Ref.",
        marker="+",
        alpha=0.8,
        s=100,  # 适当增大点大小
        c="k",
        # edgecolors="w",  # 白色边缘，使点更清晰
        # linewidths=0.2,
    )
    # ax0.set_ylabel(rf"Estimated ${param_name}$")
    # ax0.set_xlabel(rf"Reference ${param_name}$")
    # ax0.set_title(f"Scenairo {scenario} case {case_name} ({param_name})")

    # 添加45度参考线
    ax0.plot(
        xlim_all,
        ylim_all,
        color="black",
        linestyle="--",
    )

    # x, y轴范围
    ax0.set_xlim(xlim_all)
    ax0.set_ylim(ylim_all)
    # 绘制拟合线
    z_all = np.polyfit(refs_all, invs_all, 1)
    p = np.poly1d(z_all)
    invs_all_fit = p(refs_all)
    ax0.plot(xlim_all, [p(x) for x in xlim_all], color="red", linestyle="-")

    # 计算R2
    # ssim_normalized_all = calculate_ssim_skimage(refs_all, invs_all, normalize=True)
    # ssim_raw_all = calculate_ssim_skimage(refs_all, invs_all)

    r2_all = r2_score(invs_all, invs_all_fit)
    mse_all = mean_squared_error(refs_all, invs_all)
    rmse_all = np.sqrt(mse_all)
    mbe_all = np.mean(invs_all - refs_all)

    # 将拟合曲线表达式与范数，R2值添加到图中
    if z_all[1] >= 0:
        text_func_all = (
            r"$y = $" + f"{z_all[0]:.2f}" + r"$x + $" + f"{np.abs(z_all[1]):.2f}\n"
        )
    else:
        text_func_all = (
            r"$y = $" + f"{z_all[0]:.2f}" + r"$x - $" + f"{np.abs(z_all[1]):.2f}\n"
        )

    fig_ind = scenario_case["fig_index"]

    text_content_all = (
        text_func_all
        + r"$R^2$"
        + f": {r2_all:.2f}\n"  # R²使用LaTeX格式
        + r"$MBE$"
        + f"   : {mbe_all:.2f} "
        # + rf"$({unit})^2$"
        + "\n"
        + r"$RMSE$"
        + f": {rmse_all:.2f} "
        # + rf"$({unit})$"
        # + "\n"
        # + r"$SSIM_{raw}$"
        # + f" : {ssim_raw_all:.2f}"
        # + "\n"
        # + r"$SSIM_{norm}$"
        # + f": {ssim_normalized_all:.2f}"
    )
    if show_fig_indx:
        ax0.text(
            0.15,  # x坐标 (0是最左，1是最右)
            0.95,  # y坐标 (0是最下，1是最顶)
            text_content_all,
            transform=ax0.transAxes,  # 使用坐标轴坐标系统
            verticalalignment="top",  # 文本顶部对齐
            horizontalalignment="left",  # 文本左对齐
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.8
            ),
            fontsize=30,
        )
        ax0.text(
            0.02,  # x坐标 (0是最左，1是最右)
            0.98,  # y坐标 (0是最下，1是最顶)
            f"({fig_ind})",
            transform=ax0.transAxes,  # 使用坐标轴坐标系统
            verticalalignment="top",  # 文本顶部对齐
            horizontalalignment="left",  # 文本左对齐
            fontsize=30,
        )
    else:
        ax0.text(
            text_xy[0],  # x坐标 (0是最左，1是最右)
            text_xy[1],  # y坐标 (0是最下，1是最顶)
            text_content_all,
            transform=ax0.transAxes,  # 使用坐标轴坐标系统
            verticalalignment="top",  # 文本顶部对齐
            horizontalalignment="left",  # 文本左对齐
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.8
            ),
            fontsize=30,
        )

    ax0.set_aspect("equal")
    plt.tight_layout(
        pad=0.1,  # 图形边距
        w_pad=0.1,  # 水平间距
        h_pad=0.1,  # 垂直间距
        # rect=[0, 0, 1, 1]  # 调整区域 [left, bottom, right, top]
    )
    # 添加图例
    # ax0.legend(["Est. vs Ref.", r"$y=x$", "Fitted line"])

    scenario = scenario_case["scenario"]
    case_name = scenario_case["case"]

    if save:

        plt.savefig(
            f"{save_path}/{fig_ind}_Scenario_{scenario}_{case_name}_ref_est_tt.png",
            dpi=300,
            transparent=True,  # 背景透明
            edgecolor="none",
        )

        plt.savefig(
            f"{save_path}/{fig_ind}_Scenario_{scenario}_{case_name}_ref_est_tt.svg",
            format="svg",
            bbox_inches="tight",  # 紧密边界
            pad_inches=0.1,  # 内边距
            transparent=True,  # 背景透明
            # facecolor="white",
        )  # 背景颜色
    plt.show()
