from plotFunc.step_function import *


def visualize_combined_step_test(
    x_range=(-2, 3),
    num_points=1000,
    func_list=None,
    operation=None,
    title="Combined Step Function",
    xlabel="x",
    ylabel="f(x)",
    grid=True,
    figsize=(10, 6),
    use_smooth_step=False,
):
    """
    可视化组合后的阶跃函数

    Parameters:
    -----------
    x_range : tuple, optional
        x轴范围，默认为(-2, 3)
    num_points : int, optional
        采样点数，默认为1000
    func_list : list, optional
        阶跃函数列表
    operation : callable, optional
        组合操作函数
    title : str, optional
        图表标题
    xlabel : str, optional
        x轴标签，默认为"x"
    ylabel : str, optional
        y轴标签，默认为"f(x)"
    grid : bool, optional
        是否显示网格，默认为True
    figsize : tuple, optional
        图表大小，默认为(10, 6)
    use_smooth_step : bool, optional
        是否使用基于过渡区宽度的平滑阶跃函数，默认为False

    Returns:
    --------
    tuple
        (figure, axes)
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = combine_step_functions(x, func_list, operation, use_smooth_step)

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

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, linewidth=2, color="k")
    ax.set_title(title)
    ax.set_xlabel(rf"{xlabel}")  # , fontweight="bold")
    ax.set_ylabel(rf"{ylabel}")  # , fontweight="bold")
    if grid:
        ax.grid(True, alpha=1)

    # for item in func_list:
    #     if isinstance(item, dict):
    #         if use_smooth_step:
    #             center = item.get("center", 0)
    #             transition_width = item.get("transition_width", 0.1)
    #             # ax.axvline(x=center, color="red", linestyle="--", alpha=0.5)
    #             # if transition_width > 0:
    #             #     ax.axvspan(
    #             #         center - transition_width / 2,
    #             #         center + transition_width / 2,
    #             #         alpha=0.1,
    #             #         color="green",
    #             #     )
    #         else:
    #             step_point = item.get("step_point", 0)
    #             # ax.axvline(x=step_point, color="red", linestyle="--", alpha=0.5)
    ax.set_xlim(x_range)
    ax.set_yticks(np.arange(0, 1.1, 0.5))
    ax.set_ylim(-0.05, 1.1)
    ax.set_aspect("equal")

    import os
    os.makedirs("./figure_table_output/Figure/FigureS1", exist_ok=True)
    plt.savefig(
        "./figure_table_output/Figure/FigureS1/FigureS1_step.svg",
        format="svg",
        bbox_inches="tight",  # 紧密边界
        pad_inches=0.1,  # 内边距
        transparent=True,  # 背景透明
        facecolor="white",
    )

    return fig, ax


func_list_smooth2 = [
    {"center": 0.5, "transition_width": 1, "value_before": 0, "value_after": 1},
    {"center": 1.5, "transition_width": 1, "value_before": 0, "value_after": 1},
]
operation = lambda results: results[0] * (1 - results[1])

visualize_combined_step_test(
    x_range=(0, 5),
    func_list=func_list_smooth2,
    operation=operation,
    title="",
    use_smooth_step=True,
    xlabel="Time ($s$)",
    ylabel="Normalized rate",
    # grid=False,
    figsize=(15, 5),
)
# plt.show()
