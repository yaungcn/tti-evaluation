# Code and data for "Eikonal-Based Hydraulic Travel Time Inversion for Aquifer Reconstruction: Performance Evaluation and Surrounding Medium Effects" submitted to Water Resources Research.

> Supporting code and data for generating figures in the manuscript on travel-time tomography using forward modeling and SIRT inversion.

**Author:** Song, Yang — soongyaung@hhu.edu.cn
**Date:** 2026-06

## Directory Structure

```
Code_Figures_Table/
├── Figure2b-e-f.py                          # Statistical comparison of True D vs Est D (2a/2b/2c)
├── Figure3_combined_ref.py                  # Multi-panel scatter: True vs Est D (8 heterogeneous cases)
├── Figure4_combined_boxplot.py              # Boxplots of RMSE, MBE, r, Slope for all scenarios
├── Figure5a-g.py                            # Travel time comparison views for scenario 2a case 2
├── Figure6a-c.py                            # Travel time comparison views for scenario 3a
├── Figure7_combined_v2.py                   # Travel time contour maps with varying surrounding medium
├── FigureS1_stepfunction.py                 # Step function visualization (hard/smooth)
├── FigureS2_combined_traveltime_hetero_comparison_sirt.py  # Combined SIRT travel time comparison
├── FigureS3_combined_traveltime_hetero_comparison.py      # Travel time comparison (same layout as Figure3)
├── TableS2_dataset.py                       # Compute statistics and export TableS2.xlsx
│
├── dataFunc/               # Data I/O and processing: traveltime, statistical processing
│   ├── dataProcess.py      #   Read/write inversion results, mesh, traveltime CSVs
│   ├── statisticalProcess.py#  Statistical indicators (RMSE, MBE, R², SSIM)
│   └── traveltimeProcess.py#  Traveltime level classification (L0–L4)
├── geostatFunc/            # Geostatistical analysis utilities
│   ├── gstat.py            #   Variogram modeling (gstools)
│   └── statistics.py       #   DataFrame helpers, boxplot data setup for cases 2a/2b/2c
├── inverseFunc/            # Inversion utilities
│   └── pgSIRT.py           #   SIRT inversion via pyGIMLi's TravelTimeManager
├── plotFunc/               # Plotting utilities
│   ├── boxplot.py          #   Boxplot rendering (setup, 2-case, 4-case, legend)
│   ├── comparisonPlot.py   #   True-vs-est scatter with R²/RMSE annotations
│   ├── Figure6_plot.py     #   All plotting for Figure 6a-c
│   ├── plot_mesh.py        #   Mesh file parsing and visualization
│   ├── plot_travel_time.py #   Travel time map loading and contour plotting
│   ├── step_function.py    #   Hard/smooth step function primitives
│   └── traveltimeScatter.py#   Level-based traveltime scatter plots
├── data/                   # Input data (scenario-based directory structure)
│   ├── dataset_mesh/       #   Mesh file (net-20x20-v2.txt)
│   ├── dataset_scenarios/  #   Scenario data: 1a/ (mean field), 2a/2b/2c/ (heterogeneous), 3a/ (field study)
│   │   └── {scenario}/{case_name}/lnKs/    # ln(K) fields (hetero_field, homo_field, invsResult)
│   │   └── {scenario}/{case_name}/model_export/      # SIRT travel time data (CSV)
│   │   └── {scenario}/{case_name}/model_export_inv/  # SIRT inverted travel time data (CSV)
│   └── dataset_sirt_traveltime/  # Precomputed SIRT eikonal travel time (.npy)
└── figure_table_output/    # Generated outputs
    ├── Figure/             #   Figures organized by number (Figure2–7, FigureS1–S3)
    └── Tables/             #   Generated tables (TableS2_stats_dataset.xlsx)
```

## Figure Scripts Overview

| Script | Output | Description |
|--------|--------|-------------|
| `Figure2b-e-f.py` | `figure_table_output/Figure/Figure2/` | True vs Estimated ln(D) scatter with regression, sub-figures b/e/f |
| `Figure3_combined_ref.py` | `figure_table_output/Figure/Figure3/` | 8-panel combined scatter for all heterogeneous cases |
| `Figure4_combined_boxplot.py` | `figure_table_output/Figure/Figure4/` | Boxplots of error metrics per scenario |
| `Figure5a-g.py` | `figure_table_output/Figure/Figure5/` | Eikonal travel time (a) + simulated travel time (b–g) |
| `Figure6a-c.py` | `figure_table_output/Figure/Figure6/` | Travel time views for scenario 3a (full + zooms) |
| `Figure7_combined_v2.py` | `figure_table_output/Figure/Figure7/` | Travel time contour maps for varying medium |
| `FigureS1_stepfunction.py` | `figure_table_output/Figure/FigureS1/` | Step function illustration |
| `FigureS2_combined_traveltime_hetero_comparison_sirt.py` | `figure_table_output/Figure/FigureS2/` | Combined SIRT travel time comparison |
| `FigureS3_*.py` | `figure_table_output/Figure/FigureS3/` | Heterogeneous travel time comparison |
| `TableS2_dataset.py` | `figure_table_output/Tables/TableS2_stats_dataset.xlsx` | Statistical indicators (RMSE, MBE, R², etc.) |

## Scenarios

| Case | Description | Sub-cases |
|------|-------------|-----------|
| 1a | Mean (homogeneous) field | mean = –9.2, –11.5 |
| 2a | Heterogeneous (varying mean) | mean\_exp = –9.2, –11.5 |
| 2b | Heterogeneous (varying variance) | var\_lnK = 2, 4, 8 |
| 2c | Heterogeneous (varying correlation length) | cor\_lnK = 2, 6, 8 |
| 3a | Field study (5 mean values) | –13.8, –11.5, –9.2, –6.9, –4.6 |

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Place the dataset under `./data/` with the following structure:
   ```
   data/
   ├── dataset_mesh/                # net-20x20-v2.txt
   ├── dataset_scenarios/           # {1a,2a,2b,2c,3a}/{case_name}/...
   └── dataset_sirt_traveltime/     # {2a,2b,2c}/{case_name}/*.npy
   ```
3. Run any figure script directly:
   ```
   python Figure2b-e-f.py
   python Figure3_combined_ref.py
   ...
   ```

All figure scripts read data from `./data/` and save outputs to `./figure_table_output/Figure/`. Table scripts output to `./figure_table_output/Tables/`.

## Dependencies

- `numpy`, `matplotlib`, `pandas`, `scipy`, `scikit-learn`
- `pygimli` (SIRT inversion)
- `gstools` (variogram modeling)
- `toml` (field parameter parsing)
- `brewer2mpl` (color palettes)

## Notes

- Figure scripts contain Chinese comments documenting scenario parameters and workflow steps.

## License

MIT License

Copyright (c) 2026 Song, Yang
