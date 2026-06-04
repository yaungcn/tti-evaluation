# %% [markdown]
# # Figure 6a-c: Traveltime comparison — full view and two zoom-in views

# %%
from plotFunc.Figure6_plot import (
    load_figure6_data,
    _get_line_styles,
    _get_legend_labels,
    plot_figure6a,
    plot_figure6b,
    plot_figure6c,
    plot_figure6_combined,
)

LANGUAGE = "en"
SAVE_DIR = "./figure_table_output/Figure/Figure6"

# %%
length_path_2, ttdata, correct_ttdata = load_figure6_data(
    workpath="./data",
    field_name="dataset_scenarios",
    scenarios=["3a"],
    cases_name=[
        "mean_-13.8",
        "mean_-11.5",
        "mean_-9.2",
        "mean_-6.9",
        "mean_-4.6",
    ],
)

line_styles = _get_line_styles()
legend_labels = _get_legend_labels(LANGUAGE)

# %%
plot_figure6a(length_path_2, correct_ttdata, line_styles, legend_labels,
              language=LANGUAGE, save_dir=SAVE_DIR)

plot_figure6b(length_path_2, correct_ttdata, line_styles, legend_labels,
              language=LANGUAGE, save_dir=SAVE_DIR)

plot_figure6c(length_path_2, correct_ttdata, line_styles, legend_labels,
              language=LANGUAGE, save_dir=SAVE_DIR)

# %%
plot_figure6_combined(length_path_2, correct_ttdata, line_styles, legend_labels,
                      language=LANGUAGE, save_dir=SAVE_DIR)
