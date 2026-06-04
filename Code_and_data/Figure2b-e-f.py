# %% [markdown]
# # Figure 2: Reference field, Inversion field, and Scatter plot (no.144)
# # Output: 3 figures → image/Figure2/

# %%
from dataFunc.statisticalProcess import *
from sklearn.metrics import r2_score, mean_squared_error
from dataFunc.statisticalProcess import calculate_statiscal_indicators as csi
import numpy as np
import os

# %%
field_name = "dataset_scenarios"
scenario = "2a"
case_name = "mean_exp_-11.5"
num_case = 144

invdata_lnKs = read_refdata(
    f"./data/{field_name}/{scenario}/{case_name}/lnKs/invsResult"
)
refdata_lnKs = read_refdata(
    f"./data/{field_name}/{scenario}/{case_name}/lnKs/hetero_field"
)
ref_Ks = np.array(refdata_lnKs.data)
inv_Ks = np.array(invdata_lnKs.data)

savepath = "./figure_table_output/Figure/Figure2"
os.makedirs(savepath, exist_ok=True)

# %% [markdown]
# # Figure 1: Reference field no.144 — ln(K) + ln(S_S)

# %%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_fig1_ref_lnK_lnSs(ref_D, num_case, savepath):
    x = y = np.linspace(0, 10, 10)
    well_x = [0, 0, 0, 0, 0, 10, 10, 10, 10, 10]
    well_y = [0, 2.5, 5, 7.5, 10, 0, 2.5, 5, 7.5, 10]
    source_x = [5, 5, 5, 5, 5]
    source_y = [0, 2.5, 5, 7.5, 10]

    refs_lnK = ref_D[num_case]

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )

    cmap_custom = plt.cm.RdBu_r

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    contour1 = ax1.contourf(x, y, refs_lnK, levels=20, cmap=cmap_custom)
    ax1.scatter(
        well_x,
        well_y,
        c="white",
        marker="o",
        s=80,
        edgecolors="black",
        linewidths=1,
        zorder=10,
    )
    ax1.scatter(
        source_x,
        source_y,
        c="white",
        marker="X",
        s=80,
        edgecolors="black",
        linewidths=1,
        zorder=10,
    )
    ax1.set_aspect("equal")
    ax1.set_title(rf"Ref. Hetero. $ln(K)$ (no.{num_case})")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    cbar1 = plt.colorbar(contour1, ax=ax1, shrink=0.9, pad=0.05)
    cbar1.set_label(r"$ln(K)$ (m/s)")

    ref_Ss = np.ones((10, 10)) * -11.5
    contour2 = ax2.contourf(
        x, y, ref_Ss, levels=1, vmin=-12.0, vmax=-11.0, cmap=cmap_custom
    )
    ax2.scatter(
        well_x,
        well_y,
        c="white",
        marker="o",
        s=80,
        edgecolors="black",
        linewidths=1,
        zorder=10,
    )
    ax2.scatter(
        source_x,
        source_y,
        c="white",
        marker="X",
        s=80,
        edgecolors="black",
        linewidths=1,
        zorder=10,
    )
    ax2.set_aspect("equal")
    ax2.set_title(rf"Ref. Homo. $ln(S_S)$ (no.{num_case})")
    ax2.set_xlabel("X (m)")
    cbar2 = plt.colorbar(contour2, ax=ax2, shrink=0.9, pad=0.05)
    cbar2.set_ticks([-11.5])
    cbar2.set_ticklabels(["-11.5"])
    cbar2.set_label(r"$ln(S_S)$")

    plt.tight_layout(pad=0.1, w_pad=0.1)
    plt.savefig(
        f"{savepath}/Figure2_ref_lnK_lnSs_no{num_case}.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.savefig(
        f"{savepath}/Figure2_ref_lnK_lnSs_no{num_case}.svg",
        format="svg",
        bbox_inches="tight",
        pad_inches=0.1,
        transparent=True,
    )
    plt.show()
    plt.close()


plot_fig1_ref_lnK_lnSs(ref_Ks, num_case, savepath)

# %% [markdown]
# # Figure 2: Inversion field no.144 — estimated ln(K)


# %%
def plot_fig2_est_lnK(inv_D, num_case, savepath):
    x = y = np.linspace(0, 10, 10)
    well_x = [0, 0, 0, 0, 0, 10, 10, 10, 10, 10]
    well_y = [0, 2.5, 5, 7.5, 10, 0, 2.5, 5, 7.5, 10]
    source_x = [5, 5, 5, 5, 5]
    source_y = [0, 2.5, 5, 7.5, 10]

    invs = inv_D[num_case]

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )

    cmap_custom = plt.cm.RdBu_r

    fig, ax = plt.subplots(figsize=(8, 7))

    contour = ax.contourf(x, y, invs, levels=20, cmap=cmap_custom)
    ax.scatter(
        well_x,
        well_y,
        c="white",
        marker="o",
        s=80,
        edgecolors="black",
        linewidths=1,
        zorder=10,
    )
    ax.scatter(
        source_x,
        source_y,
        c="white",
        marker="X",
        s=80,
        edgecolors="black",
        linewidths=1,
        zorder=10,
    )
    ax.set_aspect("equal")
    ax.set_title(rf"Est. $ln(K)$ (no.{num_case})")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    cbar = plt.colorbar(contour, ax=ax, shrink=0.9, pad=0.05)
    cbar.set_label(r"$ln(K)$ (m/s)")

    plt.tight_layout(pad=0.1)
    plt.savefig(
        f"{savepath}/Figure2_est_lnK_no{num_case}.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.savefig(
        f"{savepath}/Figure2_est_lnK_no{num_case}.svg",
        format="svg",
        bbox_inches="tight",
        pad_inches=0.1,
        transparent=True,
    )
    plt.show()
    plt.close()


plot_fig2_est_lnK(inv_Ks, num_case, savepath)

# %% [markdown]
# # Figure 3: Reference vs Estimated ln(K) scatter plot


# %%
def plot_fig3_ref_vs_est_scatter(ref_D, inv_D, num_case, savepath):
    refs = ref_D[num_case]
    invs = inv_D[num_case]

    refs_list = refs.flatten()
    invs_list = invs.flatten()

    max_D = max(np.max(invs_list), np.max(refs_list))
    min_D = min(np.min(invs_list), np.min(refs_list))
    xlim = [int(min_D) - 1.5, int(max_D) + 1.5]

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["mathtext.fontset"] = "stixsans"
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

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(refs_list, invs_list, label="Ref. vs Est.", marker="+", alpha=0.8)
    ax.set_xlabel(r"Ref. $ln(K)$ (m/s)")
    ax.set_ylabel(r"Est. $ln(K)$ (m/s)")
    ax.set_title(rf"Ref. vs Est. $ln(K)$ (no.{num_case})")

    ax.plot(xlim, xlim, color="black", linestyle="--", label=r"$y=x$")
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)

    z = np.polyfit(refs_list, invs_list, 1)
    p = np.poly1d(z)
    ax.plot(xlim, [p(x) for x in xlim], color="red", linestyle="-", label="Fitted line")
    ax.set_aspect("equal")

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
    ) = csi(refs, invs)

    if z[1] >= 0:
        text_func = r"$y = $" + f"{z[0]:.2f}" + r"$x + $" + f"{np.abs(z[1]):.2f}\n"
    else:
        text_func = r"$y = $" + f"{z[0]:.2f}" + r"$x - $" + f"{np.abs(z[1]):.2f}\n"

    text_content = (
        text_func
        + r"$RMSE$"
        + f": {rmse:.2f}\n"
        + r"$MBE$"
        + f"   : {mbe:.2f}\n"
        + r"$Pearson's\ r\ $"
        + f": {r_value:.2f}\n"
        + r"$R^2$"
        + f": {r2_value:.2f}\n"
        + r"$SSIM_{global}$"
        + f": {ssim_g:.2f}\n"
        + r"$SSIM_{local}\ \ $"
        + f": {ssim_l:.2f}"
    )

    ax.text(
        0.05,
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

    ax.legend(["Ref. vs Est.", r"$y=x$", "Fitted line"])

    plt.tight_layout(pad=0.1)
    plt.savefig(
        f"{savepath}/Figure2_scatter_ref_est_lnK_no{num_case}.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.savefig(
        f"{savepath}/Figure2_scatter_ref_est_lnK_no{num_case}.svg",
        format="svg",
        bbox_inches="tight",
        pad_inches=0.1,
        transparent=True,
    )
    plt.show()
    plt.close()


plot_fig3_ref_vs_est_scatter(ref_Ks, inv_Ks, num_case, savepath)
