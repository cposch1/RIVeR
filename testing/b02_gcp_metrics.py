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
from scipy.stats import linregress
import os

# %%
# ==================================================
# USER SETTINGS
# ==================================================

cam = "ilh-cam1-pt"
direc = Path(r"C:\Users\cposch1\OneDrive - Université de Lausanne\FlowState\ch1_hydro\RIVeR\trial_res_2026-07-07_gcp_test_ih132\00_gcps1-5\gcps")
temp_dat = 'KAN_U_hour_new.csv'
start_date = pd.to_datetime("2025-07-01 00:00")
end_date   = pd.to_datetime("2025-07-25 00:00")

# ==================================================
# LOAD GCPS DATA (Points 1-4)
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
df_gcps["datetime"] = pd.to_datetime(df_gcps["datetime"]).dt.normalize()


# ==================================================
# LOAD STAKE POINT DATA (Point 5 only)
# ==================================================

cam_dir = direc / cam
records = []

for f in sorted(cam_dir.glob(f"{cam}_stake_point_*.csv")):

    try:
        timestamp = f.stem.split("_")[-2] + "_" + f.stem.split("_")[-1]
        dt = pd.to_datetime(timestamp, format="%Y%m%d_%H%M%S")

        df_tmp = pd.read_csv(f, header=None)

        records.append({
            "datetime": dt,
            "5_x": df_tmp.iloc[0, 0],
            "5_y": df_tmp.iloc[0, 1],
        })

    except Exception as e:
        print(f"Skipping {f.name}: {e}")

df_stake = (
    pd.DataFrame(records)
    .dropna()
    .sort_values("datetime")
    .reset_index(drop=True)
)
df_stake["datetime"] = pd.to_datetime(df_stake["datetime"]).dt.normalize()


# ==================================================
# LOAD TEMP DATA
# ==================================================
kan_u = pd.read_csv(temp_dat, index_col=0)
df_kanu = (
    kan_u[["t_u"]]
    .reset_index()
    .rename(columns={kan_u.index.name or kan_u.columns[0]: "datetime"})
)

df_kanu["datetime"] = pd.to_datetime(df_kanu["datetime"])
df_kanu["PDD"] = df_kanu["t_u"].clip(lower=0)
df_kanu["cum_PDD"] = (
    df_kanu["t_u"]
    .clip(lower=0)
    .fillna(0)
    .cumsum()
)


# ==================================================
# COMBINE GCPS AND TEMP DATA
# ==================================================
df_comb = pd.merge(
    df_gcps,
    df_kanu,
    on="datetime",
    how="inner"
)

# %% [markdown]
# # 1. GCP points metrics

# %% [markdown]
# ## Spatio-temporal scatters

# %%
# Scatter + regression

x_var = "datetime"
y_var = "y"

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i in range(1, 5):
    ax = axes[i-1]

    if x_var == "datetime":
        x = df_gcps[x_var].values
    elif len(x_var) > 1:
        if x_var.startswith("5"):
            x = df_stake[x_var].values
        else:
            x = df_gcps[x_var].values
        #ax.set_xlim(300,550)
    else:
        x = df_gcps[f"{i}_{x_var}"].values
        #ax.set_xlim(300,550)

    if y_var == "datetime":
        y = df_gcps[y_var].values
    elif len(y_var) > 1:
        if y_var.startswith("5"):
            y = df_stake[y_var].values
        else:
            y = df_gcps[y_var].values
        #ax.set_ylim(300,550)
    else:
        y = df_gcps[f"{i}_{y_var}"].values
        #ax.set_ylim(300,550)
    

    # Scatter
    ax.scatter(x, y, color="blue")
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.grid()
    ax.tick_params(axis='x', rotation=90)
    ax.set_title(f"GCP {i}")

    # Regression
    

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

# %% [markdown]
# ## PDD(H) scatter

# %%
# Scatter + regression

x_var = "cum_PDD"
y_var = "y"

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i in range(1, 5):
    ax = axes[i-1]
    x = df_comb[x_var].values
    y = df_comb[f"{i}_{y_var}"].values
    
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
df_m = pd.merge(
    df_stake,
    df_kanu,
    on="datetime",
    how="inner"
)

# %%
df_m

# %% [markdown]
# # 2. Stake point metrics

# %% [markdown]
# ## PDD(H) scatter

# %%
x_var = "cum_PDD"
y_var = "5_y"

x = df_m[x_var]
y = df_m[y_var]

plt.figure(figsize=(5,5))
plt.scatter(x, y)

plt.xlabel(x_var)
plt.ylabel(y_var)
plt.grid(True)

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

    plt.plot(
        x_fit,
        y_fit,
        color="red",
        linewidth=2,
        label=(
            f"y = {b:.3f}x + {c:.3f}\n"
            f"r = {r:.3f}"
        )
    )

    plt.legend()

plt.show()

# %% [markdown]
# # 3. Modell GCP img coordinates

# %%
model_dir = Path(r"C:\Users\cposch1\OneDrive - Université de Lausanne\FlowState\ch1_hydro\RIVeR\trial_res_2026-07-07_gcp_test_ih132\02_gcp_modelling")
mod_in = model_dir / "input"
mod_out = model_dir / "output"

cam = "ilh-cam1-pt"
direc = Path(r"C:\Users\cposch1\OneDrive - Université de Lausanne\FlowState\ch1_hydro\RIVeR\trial_res_2026-07-07_gcp_test_ih132\00_gcps1-5\gcps")

# %%
# ==================================================
# LOAD GCPS DATA (Points 1-4)
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
df_gcps["datetime"] = pd.to_datetime(df_gcps["datetime"]).dt.normalize()

# %%
df_gcps

# %%
# ==================================================
# LOAD STAKE POINT DATA (Point 5 only)
# ==================================================
cam_dir = direc / cam
records = []

for f in sorted(cam_dir.glob(f"{cam}_stake_point_*.csv")):

    try:
        timestamp = f.stem.split("_")[-2] + "_" + f.stem.split("_")[-1]
        dt = pd.to_datetime(timestamp, format="%Y%m%d_%H%M%S")

        df_tmp = pd.read_csv(f, header=None)

        records.append({
            "datetime": dt,
            "5_x": df_tmp.iloc[0, 0],
            "5_y": df_tmp.iloc[0, 1],
        })

    except Exception as e:
        print(f"Skipping {f.name}: {e}")

df_stake = (
    pd.DataFrame(records)
    .dropna()
    .sort_values("datetime")
    .reset_index(drop=True)
)
df_stake["datetime"] = pd.to_datetime(df_stake["datetime"]).dt.normalize()

# %%
df_stake

# %%
df_5gcp = pd.merge(
    df_gcps,
    df_stake,
    on="datetime",
    how="inner"
)

# %%
df_5gcp

# %%
# define lin regression
results = []

for gcp_id in range(1, 5):
    y = df_5gcp[f"{gcp_id}_y"].values
    x = df_5gcp["5_y"].values

    mask = np.isfinite(x) & np.isfinite(y)

    if np.sum(mask) > 1:
        result = linregress(x[mask], y[mask])

        results.append({
            "gcp_id": gcp_id,
            "slope": result.slope,
            "intercept": result.intercept,
            "rvalue": result.rvalue,
            "pvalue": result.pvalue,
            "stderr": result.stderr,
            "intercept_stderr": result.intercept_stderr
        })

df_lin = pd.DataFrame(results)


csv_path = mod_in/ "df_lin.csv"

with open(csv_path, "w", encoding="utf-8", newline="") as f:
    f.write(f"direc,{direc}\n")  # first line
    df_lin.to_csv(f, index=False)  # dataframe starts on second line

# %%
df_lin

# %%
# --------------------------------------------------------------
# Read fixed x coordinates from first available GCP file
# --------------------------------------------------------------

first_gcp_file = next(mod_in.glob("*_gcps_img_*.csv"))
print(first_gcp_file)

gcp_ref = pd.read_csv(
    first_gcp_file,
    header=None
)

fixed_x = {
    gcp_id: gcp_ref.iloc[gcp_id - 1, 0]
    for gcp_id in range(1, 5)
}

print("Using fixed x coordinates:", fixed_x)

# --------------------------------------------------------------
# Model GCP image coordinates from stake point
# Model ALL stake files
# --------------------------------------------------------------

stake_files = sorted(
    mod_in.glob("*_stake_point_*.csv")
)

for stake_file in stake_files:

    stake_df = pd.read_csv(
        stake_file,
        header=None
    )

    x5 = float(stake_df.iloc[0, 0])
    y5 = float(stake_df.iloc[0, 1])

    output_rows = []

    for gcp_id in range(1, 5):

        x = fixed_x[gcp_id]

        reg = df_lin.loc[
            df_lin["gcp_id"] == gcp_id
        ].iloc[0]

        y_new = (
            reg["slope"] * y5
            + reg["intercept"]
        )

        output_rows.append([
            int(round(x)),
            int(round(y_new))
        ])

    out_name = stake_file.name.replace(
        "_stake_point_",
        "_gcps_img_"
    )

    out_file = mod_out / out_name

    pd.DataFrame(output_rows).to_csv(
        out_file,
        header=False,
        index=False
    )

    print(f"Written: {out_file.name}")

# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.stats import linregress

# =====================================================
# USER SETTINGS
# =====================================================

x_var = "5_y"      # "datetime" or "5_y"
coord = "y"             # "x" or "y"

# =====================================================
# PREP STAKE DATA
# =====================================================

records = []

stake_files = sorted(
    mod_in.glob("*_stake_point_*.csv")
)

for stake_file in stake_files:

    m = re.search(
        r"(\d{8})_(\d{6})",
        stake_file.name
    )

    if m is None:
        continue

    dt = pd.to_datetime(
        m.group(1) + m.group(2),
        format="%Y%m%d%H%M%S"
    )

    stake_df = pd.read_csv(
        stake_file,
        header=None
    )

    records.append(
        {
            "datetime": dt,
            "5_x": stake_df.iloc[0, 0],
            "5_y": stake_df.iloc[0, 1],
        }
    )

df_stake_mod = (
    pd.DataFrame(records)
    .sort_values("datetime")
    .reset_index(drop=True)
)

df_stake_mod["datetime"] = (
    pd.to_datetime(df_stake_mod["datetime"])
    .dt.normalize()
)

print(len(df_stake_mod))
print(df_stake_mod.head())
print(df_stake_mod.tail())

stake_tmp = df_stake_mod.copy()

# =====================================================
# PLOT
# =====================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10)
)

axes = axes.flatten()

for gcp_id in range(1, 5):

    # -------------------------------------------------
    # Read modelled GCP files
    # -------------------------------------------------

    records = []

    for f in sorted(
        mod_out.glob("*_gcps_img_*.csv")
    ):

        m = re.search(
            r"(\d{8})_(\d{6})",
            f.name
        )

        if m is None:
            continue

        dt = pd.to_datetime(
            m.group(1) + m.group(2),
            format="%Y%m%d%H%M%S"
        )

        df = pd.read_csv(
            f,
            header=None
        )

        records.append(
            {
                "datetime": dt,
                "x": df.iloc[gcp_id - 1, 0],
                "y": df.iloc[gcp_id - 1, 1],
            }
        )

    df_plot = pd.DataFrame(records)

    if len(df_plot) == 0:
        print(
            f"No records found for "
            f"GCP {gcp_id}"
        )
        continue

    # -------------------------------------------------
    # Normalize dates
    # -------------------------------------------------

    df_plot["datetime"] = (
        pd.to_datetime(df_plot["datetime"])
        .dt.normalize()
    )

    df_plot = (
        df_plot
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # -------------------------------------------------
    # Join with stake data
    # -------------------------------------------------

    merged = (
        pd.merge(
            df_plot,
            stake_tmp[["datetime", "5_y"]],
            on="datetime",
            how="inner"
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    print(
        f"GCP {gcp_id}: "
        f"df_plot={len(df_plot)}, "
        f"stake={len(stake_tmp)}, "
        f"merged={len(merged)}"
    )

    if len(merged) == 0:
        continue

    ax = axes[gcp_id - 1]

    # -------------------------------------------------
    # X-axis
    # -------------------------------------------------

    if x_var == "datetime":

        x = merged["datetime"].values

    elif x_var == "5_y":

        x = merged["5_y"].values

    else:

        raise ValueError(
            f"Unknown x_var: {x_var}"
        )

    # -------------------------------------------------
    # Y-axis
    # -------------------------------------------------

    y = merged[coord].values

    # -------------------------------------------------
    # Scatter
    # -------------------------------------------------

    ax.scatter(
        x,
        y,
        color="blue"
    )

    # -------------------------------------------------
    # Time series
    # -------------------------------------------------

    if x_var == "datetime":


        ax.tick_params(
            axis="x",
            rotation=90
        )

    # -------------------------------------------------
    # Regression
    # -------------------------------------------------

    else:

        mask = (
            np.isfinite(x)
            &
            np.isfinite(y)
        )

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

    # -------------------------------------------------
    # Formatting
    # -------------------------------------------------

    ax.set_title(
        f"GCP {gcp_id}"
    )

    ax.set_xlabel(
        x_var
    )

    ax.set_ylabel(
        coord
    )

    ax.grid(True)

plt.tight_layout()
plt.show()

# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.stats import linregress

# =====================================================
# LOAD MODELLED GCP FILES
# =====================================================

records = []

for f in sorted(mod_out.glob("*_gcps_img_*.csv")):

    m = re.search(
        r"(\d{8})_(\d{6})",
        f.name
    )

    if m is None:
        continue

    dt = pd.to_datetime(
        m.group(1) + m.group(2),
        format="%Y%m%d%H%M%S"
    )

    df_tmp = pd.read_csv(
        f,
        header=None
    )

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

df_gcps_mod = (
    pd.DataFrame(records)
    .sort_values("datetime")
    .reset_index(drop=True)
)

df_gcps_mod["datetime"] = (
    pd.to_datetime(df_gcps_mod["datetime"])
    .dt.normalize()
)

print("Modelled GCP rows:", len(df_gcps_mod))
print(df_gcps_mod.head())

# =====================================================
# LOAD TEMPERATURE DATA
# =====================================================

temp_dat = 'KAN_U_hour_new.csv'
kan_u = pd.read_csv(
    temp_dat,
    index_col=0
)

df_kanu = (
    kan_u[["t_u"]]
    .reset_index()
)

df_kanu.columns = [
    "datetime",
    "t_u"
]

df_kanu["datetime"] = pd.to_datetime(
    df_kanu["datetime"]
).dt.normalize()

df_kanu["PDD"] = (
    df_kanu["t_u"]
    .clip(lower=0)
)

df_kanu["cum_PDD"] = (
    df_kanu["t_u"]
    .clip(lower=0)
    .fillna(0)
    .cumsum()
)

print(df_kanu.head())

# =====================================================
# MERGE MODELLED GCPS AND PDD DATA
# =====================================================

df_comb_mod = pd.merge(
    df_gcps_mod,
    df_kanu,
    on="datetime",
    how="inner"
)

print("Merged rows:", len(df_comb_mod))
print(df_comb_mod.head())

# =====================================================
# SCATTER + REGRESSION
# =====================================================

x_var = "cum_PDD"
y_var = "y"

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10)
)

axes = axes.flatten()

for i in range(1, 5):

    ax = axes[i - 1]

    x = df_comb_mod[x_var].values
    y = df_comb_mod[f"{i}_{y_var}"].values

    # Scatter
    ax.scatter(
        x,
        y,
        color="blue"
    )

    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.grid(True)
    ax.tick_params(
        axis="x",
        rotation=90
    )

    ax.set_title(
        f"Modelled GCP {i}"
    )

    # Regression
    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

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

        y_fit = (
            b * x_fit
            + c
        )

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
