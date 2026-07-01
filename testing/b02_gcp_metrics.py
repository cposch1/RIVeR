# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ==================================================
# USER SETTINGS
# ==================================================

cam = "ilh-cam1-pt"
direc = Path(r"C:\Users\cposch1\OneDrive - Université de Lausanne\FlowState\ch1_hydro\trial_res_2026-06-25_gcp_test\y_slide\gcps")
start_date = pd.to_datetime("2025-07-01 00:00")
end_date   = pd.to_datetime("2025-07-25 00:00")

# ==================================================
# LOAD GCPS DATA
# ==================================================

cam_dir = direc / cam
records = []

for f in sorted(cam_dir.glob(f"{cam}_gcps_img_*.csv")):

    try:
        timestamp = f.stem.split("_")[-2] + "_" + f.stem.split("_")[-1]
        dt = pd.to_datetime(timestamp, format="%Y%m%d_%H%M%S")

        df_tmp = pd.read_csv(f, header=None)

        records.append({
            "datetime": dt,
            "1_x": df_tmp.iloc[0, 0],
            "1_y": df_tmp.iloc[0, 1],
            "2_x": df_tmp.iloc[1, 0],
            "2_y": df_tmp.iloc[1, 1],
            "3_x": df_tmp.iloc[2, 0],
            "3_y": df_tmp.iloc[2, 1],
            "4_x": df_tmp.iloc[3, 0],
            "4_y": df_tmp.iloc[3, 1],
        })

    except Exception as e:
        print(f"Skipping {f.name}: {e}")

df_gcps = (
    pd.DataFrame(records)
    .dropna()
    .sort_values("datetime")
    .reset_index(drop=True)
)

# %%
# Scatter + regression

x_var = "2_y"
y_var = "y"

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i in range(1, 5):
    ax = axes[i-1]

    if x_var == "datetime":
        x = df_gcps[x_var].values
    elif len(x_var) > 1:
        x = df_gcps[x_var].values
        ax.set_xlim(300,550)
    else:
        x = df_gcps[f"{i}_{x_var}"].values
        ax.set_xlim(300,550)

    if y_var == "datetime":
        y = df_gcps[y_var].values
    elif len(y_var) > 1:
        y = df_gcps[y_var].values
        ax.set_ylim(300,550)
    else:
        y = df_gcps[f"{i}_{y_var}"].values
        ax.set_ylim(300,550)
    

    # Scatter
    ax.scatter(x, y, color="blue")
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.grid()
    ax.tick_params(axis='x', rotation=90)
    ax.set_title(f"GCP {i}")

    # Regression
    from scipy.stats import linregress

    if x_var != "datetime" and y_var != "datetime":
    
        mask = np.isfinite(x) & np.isfinite(y)
    
        if np.sum(mask) > 1:
    
            result = linregress(
                x[mask],
                y[mask]
            )
    
            b = result.slope
            c = result.intercept
            r = result.rvalue
    
            x_fit = np.linspace(
                np.min(x[mask]),
                np.max(x[mask]),
                100
            )
    
            y_fit = b * x_fit + c
    
            ax.plot(
                x_fit,
                y_fit,
                color="red",
                linewidth=2,
                label=(
                    f"y = {b:.3f}x + {c:.3f}\n"
                    f"r = {r:.3f}"
                )
            )
    
            ax.legend()

plt.tight_layout()
plt.show()

# %%
df_gcps

# %%
