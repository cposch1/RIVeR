# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from river.config import gcps_dir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy.stats import linregress
import os
import shutil

# %% [markdown]
# ## Load data for metrics

# %%
# ==================================================
# USER SETTINGS
# ==================================================

cam = "ilh-cam1-pt"
direc = gcps_dir
temp_dat = direc / 'KAN_U_hour_new.csv'

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
            "timestamp": dt,  # exact frame time, used for pairing
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

records = []

for f in sorted(cam_dir.glob(f"{cam}_stake_point_*.csv")):

    try:
        timestamp = f.stem.split("_")[-2] + "_" + f.stem.split("_")[-1]
        dt = pd.to_datetime(timestamp, format="%Y%m%d_%H%M%S")

        df_tmp = pd.read_csv(f, header=None)

        records.append({
            "datetime": dt,
            "timestamp": dt,  # exact frame time, used for pairing
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
df_kanu["PDH"] = df_kanu["t_u"].clip(lower=0)
df_kanu["cum_PDH"] = (
    df_kanu["t_u"]
    .clip(lower=0)
    .fillna(0)
    .cumsum()
)

# Aggregate hourly temperature data to one row per day
df_kanu_daily = (
    df_kanu
    .assign(datetime=df_kanu["datetime"].dt.normalize())
    .groupby("datetime", as_index=False)
    .agg(
        PDH=("PDH", "sum"),
        cum_PDH=("cum_PDH", "last"),
    )
)


# ==================================================
# COMBINE GCPS AND TEMP DATA
# ==================================================
df_comb = pd.merge(
    df_gcps,
    df_kanu_daily,
    on="datetime",
    how="inner"
)


# ==================================================
# PAIR GCPS 1-4 WITH STAKE POINT (same frame)
# ==================================================
# The stake point has more frames than GCPs 1-4; pair them by exact frame
# timestamp so both axes come from the same frame (coverage of GCPs 1-4)
df_pair = pd.merge(
    df_gcps,
    df_stake[["timestamp", "5_x", "5_y"]],
    on="timestamp",
    how="inner"
)
print(
    f"GCP frames: {len(df_gcps)}, stake frames: {len(df_stake)}, "
    f"paired: {len(df_pair)}"
)

# %% [markdown]
# # 1. GCP points metrics

# %% [markdown]
# ## Spatio-temporal scatters

# %%
# Scatter + regression

x_var = "5_y"
y_var = "y"

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

def pair_values(var, i):
    # "datetime", "5_y", "1_x", ... taken as is; "x"/"y" = current GCP.
    # Always from df_pair, so x and y belong to the same frames.
    if var in df_pair.columns:
        return df_pair[var].values
    return df_pair[f"{i}_{var}"].values

for i in range(1, 5):
    ax = axes[i-1]

    x = pair_values(x_var, i)
    y = pair_values(y_var, i)
    #ax.set_xlim(300,550)
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
# ## PDH scatter

# %%
df_comb

# %%
# Scatter + regression

x_var = "cum_PDH"
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

# %% [markdown]
# # 2. Stake point metrics

# %% [markdown]
# ## PDH scatter

# %%
df_m = pd.merge(
    df_stake,
    df_kanu_daily,
    on="datetime",
    how="inner"
)

# %%
df_m

# %%
x_var = "cum_PDH"
y_var = "5_y"

x = df_m[x_var]
y = df_m[y_var]

plt.figure(figsize=(6,5))

# colour points by date
sc = plt.scatter(x, y, c=mdates.date2num(df_m["datetime"]), cmap="viridis")

plt.colorbar(
    sc,
    label="Date",
    ticks=mdates.AutoDateLocator(),
    format=mdates.DateFormatter("%Y-%m-%d")
)

plt.xlabel("cumulative PDH (°C)")
plt.ylabel("Y in image coordinate space (pixel)")
#plt.grid(True)

mask = np.isfinite(x) & np.isfinite(y)
    
if np.sum(mask) > 1:

    result = linregress(
        x[mask],
        y[mask]
    )

    b = result.slope
    c = result.intercept
    r = result.rvalue
    p = result.pvalue
    p_txt = "p < 0.01" if p < 0.01 else f"p = {p:.3f}"

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
            f"r = {r:.3f}, {p_txt}, n = {np.sum(mask)}"
        )
    )

    plt.legend()

plt.savefig("scat.svg")
plt.show()

# %%
x_var = "PDH"   # daily PDH (sum of hourly PDH per day)
y_var = "5_y"

x = df_m[x_var]
y = df_m[y_var]

plt.figure(figsize=(6,5))

# colour points by date
sc = plt.scatter(x, y, c=mdates.date2num(df_m["datetime"]), cmap="viridis")

plt.colorbar(
    sc,
    label="Date",
    ticks=mdates.AutoDateLocator(),
    format=mdates.DateFormatter("%Y-%m-%d")
)

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
    p = result.pvalue
    p_txt = "p < 0.01" if p < 0.01 else f"p = {p:.3f}"

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
            f"r = {r:.3f}, {p_txt}, n = {np.sum(mask)}"
        )
    )

    plt.legend()

plt.show()

# %% [markdown]
# ## Surface lowering in metres
#
# Image positions are converted to surface lowering since installation with the
# emergence calibration from a03ac (installation image: pole top + known emergence
# at a reference point, P5 or GCP1-4 -> pixels per metre along that point's column):
# lowering = (y - y_base at installation) / px_per_m.
# For a GCP reference, y is the modelled GCP position of each stake frame
# (_modelling/output); the manual picks are shown for comparison.
# The scale is assumed constant along the column (perspective changes of the scale
# below the installation surface are neglected).

# %%
# ==================================================
# LOAD EMERGENCE CALIBRATION (a03ac) AND CONVERT TO METRES
# ==================================================

emergence_file = cam_dir / f"{cam}_emergence_calibration.csv"

if not emergence_file.exists():

    print(
        f"No emergence calibration yet: {emergence_file}\n"
        "-> set it in a03ac_orthorectify_pt5.py (Pole Top panel, Save Emergence)"
    )

else:

    calib = pd.read_csv(
        emergence_file,
        dtype={"point": str, "date": str, "time": str}
    ).iloc[0]

    ref = calib["point"]
    px_per_m = float(calib["px_per_m"])
    y_install = float(calib["y_base"])
    install_day = pd.to_datetime(calib["date"], format="%Y%m%d")

    if ref == "P5":

        df_m["ref_y"] = df_m["5_y"]
        manual_ref = None

    else:

        gcp_i = int(ref[-1])

        def gcp_y(folder):
            # image y of the reference GCP per frame timestamp
            rec = {}
            for f in folder.glob(f"{cam}_gcps_img_*.csv"):
                p = f.stem.split("_")
                t = pd.to_datetime(f"{p[-2]}{p[-1]}", format="%Y%m%d%H%M%S")
                rec[t] = pd.read_csv(f, header=None).iat[gcp_i - 1, 1]
            return pd.Series(rec, dtype=float).sort_index()

        # modelled GCP position for every stake frame, manual picks for comparison
        df_m["ref_y"] = df_m["timestamp"].map(gcp_y(cam_dir / "_modelling" / "output"))
        manual_ref = gcp_y(cam_dir)

    # surface lowering since installation (positive = surface lower)
    df_m["lowering_m"] = (df_m["ref_y"] - y_install) / px_per_m

    # PDH accumulated since the installation day (same end-of-day convention as cum_PDH)
    cum_PDH_install = df_kanu_daily.set_index("datetime")["cum_PDH"].asof(install_day)
    df_m["PDH_since_install"] = df_m["cum_PDH"] - cum_PDH_install

    print(
        f"Calibration at {ref} ({calib['date']}_{calib['time']}): {calib['pole_px']} px = "
        f"{calib['emergence_m']} m -> {px_per_m:.1f} px/m ({100 / px_per_m:.2f} cm/px)"
    )

    print(df_m[["timestamp", "ref_y", "lowering_m", "PDH_since_install"]])

# %%
# ==================================================
# TIME SERIES: surface lowering at the reference point
# ==================================================

if "lowering_m" in df_m:

    plt.figure(figsize=(8, 4))

    plt.plot(
        df_m["timestamp"],
        df_m["lowering_m"],
        marker="o",
        markersize=3,
        linestyle="-",
        color="blue",
        label=f"{ref}" + (" (modelled)" if manual_ref is not None else "")
    )

    if manual_ref is not None:

        plt.plot(
            manual_ref.index,
            (manual_ref.values - y_install) / px_per_m,
            linestyle="None",
            marker="o",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor="red",
            label=f"{ref} (manual picks)"
        )

    plt.xlabel("Date")
    plt.ylabel(f"Surface lowering at {ref} since installation (m)")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.grid(True)
    plt.legend()

    plt.show()

# %%
# ==================================================
# SCATTER: surface lowering vs PDH since installation
# ==================================================
# slope = surface lowering per degree-hour; x 24 -> per degree-day

if "lowering_m" in df_m:

    x_var = "PDH_since_install"
    y_var = "lowering_m"

    x = df_m[x_var]
    y = df_m[y_var]

    plt.figure(figsize=(6,5))

    # colour points by date
    sc = plt.scatter(x, y, c=mdates.date2num(df_m["datetime"]), cmap="viridis")

    plt.colorbar(
        sc,
        label="Date",
        ticks=mdates.AutoDateLocator(),
        format=mdates.DateFormatter("%Y-%m-%d")
    )

    plt.xlabel("PDH since installation (°C)")
    plt.ylabel(f"Surface lowering at {ref} since installation (m)")

    mask = np.isfinite(x) & np.isfinite(y)

    if np.sum(mask) > 1:

        result = linregress(
            x[mask],
            y[mask]
        )

        b = result.slope
        c = result.intercept
        r = result.rvalue
        p = result.pvalue
        p_txt = "p < 0.01" if p < 0.01 else f"p = {p:.3f}"

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
                f"y = {b:.6f}x + {c:.3f}\n"
                f"r = {r:.3f}, {p_txt}, n = {np.sum(mask)}\n"
                f"{1000 * 24 * b:.1f} mm surface lowering per °C·d"
            )
        )

        plt.legend()

    plt.show()

# %% [markdown]
# # 3. Modell GCP img coordinates

# %%
model_dir = cam_dir / "_modelling"
Path(model_dir).mkdir(parents=True, exist_ok=True)
mod_in = model_dir / "input"
Path(mod_in).mkdir(parents=True, exist_ok=True)
mod_out = model_dir / "output"
Path(mod_out).mkdir(parents=True, exist_ok=True)

# %%
for csv_file in cam_dir.glob("*.csv"):
    shutil.copy(csv_file, mod_in / csv_file.name)

# %% [markdown]
# ## Load data for lin regression

# %%
# ==================================================
# LOAD pre-surface-hydrology GCPS DATA (Points 1-4)
# ==================================================
records = []

for f in sorted(mod_in.glob(f"{cam}_gcps_img_*.csv")):

    try:
        timestamp = f.stem.split("_")[-2] + "_" + f.stem.split("_")[-1]
        dt = pd.to_datetime(timestamp, format="%Y%m%d_%H%M%S")

        df_tmp = pd.read_csv(f, header=None)

        records.append({
            "datetime": dt,
            "timestamp": dt,  # exact frame time, used for pairing
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
# LOAD pre-surface-hydrology STAKE POINT DATA (Point 5 only)
# ==================================================
records = []

for f in sorted(mod_in.glob(f"{cam}_stake_point_*.csv")):

    try:
        timestamp = f.stem.split("_")[-2] + "_" + f.stem.split("_")[-1]
        dt = pd.to_datetime(timestamp, format="%Y%m%d_%H%M%S")

        df_tmp = pd.read_csv(f, header=None)

        records.append({
            "datetime": dt,
            "timestamp": dt,  # exact frame time, used for pairing
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
# COMBINE DFs
# ==================================================
# pair by exact frame timestamp (a date-only merge would pair every GCP
# frame with every stake frame of that day)
df_5gcp = pd.merge(
    df_gcps,
    df_stake[["timestamp", "5_x", "5_y"]],
    on="timestamp",
    how="inner"
)

# %%
df_gcps

# %%
df_stake

# %%
df_5gcp

# %% [markdown]
# ## Do and export lin regression

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

# %% [markdown]
# ## GCP img coordinate modelling

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

# %% [markdown]
# ## GCP model with pick errors in GCPs and stake: central fit and uncertainty envelope
#
# The regression above treats the stake picks as exact, which makes the slopes too flat.
# Here the GCP picks and the stake picks both carry the same pick error (px):
# - **central**: orthogonal (total least squares) fit of each GCP y on stake 5 y
# - **uncertainty**: parametric bootstrap with that pick error, in the July picks and in the
#   stake pick of every modelled frame -> percentiles of every modelled GCP position
# - **low / high**: the envelope edges (`env_pct`, default 16th / 84th percentile = ±1 sigma);
#   low = GCPs move less (plane higher), high = GCPs move more (plane lower)
#
# Written next to `_modelling/output` (unchanged, still the default for a03b / a04):
# `_modelling/output_central`, `output_low`, `output_high`, `_modelling/input/df_tls.csv`
# (fit and slope percentiles per GCP) and `_modelling/gcp_model_envelope.csv` (percentiles per
# frame and GCP, also of the movement since the first frame, for the vertical estimation).
# Reprocess a set with `a03b_orthorectify_auto.py ... --gcp-set central|low|high` and
# `a04_cross_section.py ... --gcp-set central|low|high`.

# %%
# ==================================================
# SETTINGS
# ==================================================
import re

env_pct = (16, 84)      # envelope written to output_low / output_high; (2.5, 97.5) = 95 %
n_draws = 2000          # bootstrap draws
rng = np.random.default_rng(42)


def tls_fit(x, y):
    # orthogonal regression: same pick error in x and y (both in px)
    xc, yc = x - x.mean(), y - y.mean()
    sxx, syy, sxy = (xc ** 2).sum(), (yc ** 2).sum(), (xc * yc).sum()
    b = (syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) / (2 * sxy)
    return b, y.mean() - b * x.mean()


# ==================================================
# CENTRAL FIT AND PICK ERROR (July picks, paired as above)
# ==================================================
x5 = df_5gcp["5_y"].values.astype(float)
tls_rows = []

for gcp_id in range(1, 5):

    y = df_5gcp[f"{gcp_id}_y"].values.astype(float)
    b, a = tls_fit(x5, y)

    # perpendicular residuals -> pick error per coordinate (px)
    d = (y - (a + b * x5)) / np.sqrt(1 + b ** 2)

    tls_rows.append({
        "gcp_id": gcp_id,
        "slope": b,
        "intercept": a,
        "slope_ols": df_lin.loc[df_lin["gcp_id"] == gcp_id, "slope"].iloc[0],
        "pick_sigma_px": np.sqrt((d ** 2).sum() / (len(d) - 2)),
    })

df_tls = pd.DataFrame(tls_rows)

# one pick error for all manual picks (GCPs and stake), pooled over the four fits
pick_sigma = np.sqrt((df_tls["pick_sigma_px"] ** 2).mean())
print(f"Pick error (pooled over GCP1-4): {pick_sigma:.2f} px")
df_tls

# %%
# ==================================================
# PARAMETRIC BOOTSTRAP -> ENVELOPE OF THE MODELLED GCP POSITIONS
# ==================================================

# all stake frames (as in the modelling above)
stake_rows = []

for stake_file in sorted(mod_in.glob("*_stake_point_*.csv")):

    m = re.search(r"(\d{8})_(\d{6})", stake_file.name)
    stake_df = pd.read_csv(stake_file, header=None)

    stake_rows.append({
        "timestamp": pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M%S"),
        "file": stake_file.name.replace("_stake_point_", "_gcps_img_"),
        "y5": float(stake_df.iloc[0, 1]),
    })

df_s = pd.DataFrame(stake_rows).sort_values("timestamp").reset_index(drop=True)
y5 = df_s["y5"].values
pct = sorted({2.5, 16, 50, 84, 97.5, *env_pct})
env_rows = []

for i, r in df_tls.iterrows():

    gcp_id, b, a = int(r["gcp_id"]), r["slope"], r["intercept"]
    y = df_5gcp[f"{gcp_id}_y"].values.astype(float)

    # picks without error: orthogonal projection of the picks onto the fitted line
    t = (x5 + b * (y - a)) / (1 + b ** 2)

    slopes = np.empty(n_draws)
    sims = np.empty((n_draws, len(y5)))

    for k in range(n_draws):

        bk, ak = tls_fit(
            t + rng.normal(0, pick_sigma, len(t)),
            a + b * t + rng.normal(0, pick_sigma, len(t))
        )
        slopes[k] = bk

        # the stake pick of every modelled frame carries the same pick error
        sims[k] = ak + bk * (y5 + rng.normal(0, pick_sigma, len(y5)))

    for p, v in zip(pct, np.percentile(slopes, pct)):
        df_tls.loc[i, f"slope_p{p:g}"] = v

    q_y = np.percentile(sims, pct, axis=0)
    q_d = np.percentile(sims - sims[:, [0]], pct, axis=0)    # movement since the first frame

    for j, s in df_s.iterrows():
        env_rows.append({
            "timestamp": s["timestamp"],
            "file": s["file"],
            "gcp_id": gcp_id,
            "y5": s["y5"],
            "y_central": a + b * s["y5"],
            **{f"y_p{p:g}": q_y[n, j] for n, p in enumerate(pct)},
            **{f"dy_p{p:g}": q_d[n, j] for n, p in enumerate(pct)},
        })

df_env = pd.DataFrame(env_rows)
df_tls.to_csv(mod_in / "df_tls.csv", index=False)
df_env.to_csv(model_dir / "gcp_model_envelope.csv", index=False)

# ==================================================
# WRITE THE THREE SETS (same format as _modelling/output)
# ==================================================
lo, hi = env_pct

for name, col in [("central", "y_central"), ("low", f"y_p{lo:g}"), ("high", f"y_p{hi:g}")]:

    out_dir = model_dir / f"output_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname, g in df_env.groupby("file"):

        g = g.sort_values("gcp_id")
        rows = [[int(round(fixed_x[gid])), int(round(v))] for gid, v in zip(g["gcp_id"], g[col])]
        pd.DataFrame(rows).to_csv(out_dir / fname, header=False, index=False)

    print(f"Written: {out_dir} ({df_env['file'].nunique()} files)")

df_tls.round(3)

# %%
# ==================================================
# PLOT: central model and envelope per GCP
# ==================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

for ax, (gcp_id, g) in zip(axes.flatten(), df_env.groupby("gcp_id")):

    r = df_tls.set_index("gcp_id").loc[gcp_id]
    ols = df_lin.set_index("gcp_id").loc[gcp_id]

    ax.fill_between(g["timestamp"], g["y_p2.5"], g["y_p97.5"],
                    color="#e7298a", alpha=0.12, lw=0, label="95 %")
    ax.fill_between(g["timestamp"], g[f"y_p{lo:g}"], g[f"y_p{hi:g}"],
                    color="#e7298a", alpha=0.3, lw=0, label=f"low / high ({lo:g}-{hi:g} %)")
    ax.plot(g["timestamp"], g["y_central"], color="#e7298a", lw=1.5, label="central")
    ax.plot(g["timestamp"], ols["intercept"] + ols["slope"] * g["y5"],
            color="tab:blue", lw=1, label=f"current output (OLS), slope {ols['slope']:.2f}")
    ax.plot(df_5gcp["timestamp"], df_5gcp[f"{gcp_id}_y"], linestyle="None", marker="o", markersize=6,
            markerfacecolor="none", markeredgecolor="red", label="manual picks")

    ax.set_title(f"GCP{gcp_id}: slope {r['slope']:.2f} "
                 f"({r[f'slope_p{lo:g}']:.2f}-{r[f'slope_p{hi:g}']:.2f}), OLS {ols['slope']:.2f}")
    ax.set_ylabel(f"GCP{gcp_id} image y (px), larger = surface lower")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(True)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

# %%
# ==================================================
# PLOT: what changed in the GCP model (documentation)
# ==================================================
# Same July picks, different fits: current OLS (blue; treats the stake picks as exact -> too flat), central fit
# with pick errors in GCP and stake (magenta, used above), reverse fit and anchoring at the first pick (not used,
# shown for comparison) and the ratio of image scales from the first-frame homography (geometry).
pairs = df_5gcp.sort_values("timestamp").reset_index(drop=True)
xs5 = pairs["5_y"].values.astype(float)


def slope_fits(x, y):
    xc, yc = x - x.mean(), y - y.mean()
    sxx, syy, sxy = (xc ** 2).sum(), (yc ** 2).sum(), (xc * yc).sum()
    ax_, ay_ = x - x[0], y - y[0]
    Sxx, Syy, Sxy = (ax_ ** 2).sum(), (ay_ ** 2).sum(), (ax_ * ay_).sum()
    tls = lambda a, b, c: (b - a + np.sqrt((b - a) ** 2 + 4 * c ** 2)) / (2 * c)
    return {"ols": sxy / sxx, "central": tls(sxx, syy, sxy), "reverse": syy / sxy, "anchored": tls(Sxx, Syy, Sxy)}


# image scale across the view from the first-frame homography (ground E/N -> image), at each GCP and at stake 5
real = pd.read_csv(cam_dir / f"{cam}_gcps_real.csv", header=None).values[:, :2].astype(float)
img0 = np.array([[pairs[f"{i}_x"].iloc[0], pairs[f"{i}_y"].iloc[0]] for i in range(1, 5)], dtype=float)
stk0 = np.array([pairs["5_x"].iloc[0], pairs["5_y"].iloc[0]], dtype=float)
wm, im0 = real.mean(0), img0.mean(0)
A = []
for (X, Y), (u, v) in zip(real - wm, img0 - im0):
    A += [[X, Y, 1, 0, 0, 0, -u * X, -u * Y, -u], [0, 0, 0, X, Y, 1, -v * X, -v * Y, -v]]
G = np.linalg.inv(np.linalg.svd(np.array(A))[2][-1].reshape(3, 3))     # image -> ground


def px_per_m_across(u, v):
    q = G @ np.array([u - im0[0], v - im0[1], 1.0])
    d = G[:, 0]
    return 1 / np.hypot((d[0] * q[2] - q[0] * d[2]) / q[2] ** 2, (d[1] * q[2] - q[1] * d[2]) / q[2] ** 2)


scale = {i: px_per_m_across(*img0[i - 1]) for i in range(1, 5)}
scale_stake = px_per_m_across(*stk0)
fits = {}
for i in range(1, 5):
    fits[i] = slope_fits(xs5, pairs[f"{i}_y"].values.astype(float))
    fits[i]["geometry"] = scale[i] / scale_stake

FIT_COL = {"ols": "tab:blue", "central": "#e7298a", "reverse": "tab:cyan", "anchored": "0.45", "geometry": "#a6611a"}
pxm_cal = float(pd.read_csv(cam_dir / f"{cam}_emergence_calibration.csv").iloc[0]["px_per_m"])
tls_tab = df_tls.set_index("gcp_id")
e4 = df_env[df_env["gcp_id"] == 4].set_index("timestamp").sort_index()
y4 = pairs["4_y"].values.astype(float)

fig = plt.figure(figsize=(19, 11))
gs = fig.add_gridspec(2, 3)


def scatter_fits(ax, i):
    y = pairs[f"{i}_y"].values.astype(float)
    xm, ym = xs5.mean(), y.mean()
    pts = pd.DataFrame({"x": xs5, "y": y}).value_counts().reset_index(name="n")
    ax.scatter(pts["x"], pts["y"], s=40 + 60 * (pts["n"] - 1), facecolors="none", edgecolors="red", lw=1.5,
               label="manual picks, July (bigger = repeated)", zorder=5)
    xx = np.array([xs5.min() - 1.5, xs5.max() + 1.5])
    f = fits[i]
    ax.plot(xx, ym + f["ols"] * (xx - xm), color=FIT_COL["ols"], lw=2, label=f"current OLS: {f['ols']:.2f}")
    ax.plot(xx, ym + f["central"] * (xx - xm), color=FIT_COL["central"], lw=2,
            label=f"central, pick errors in GCP and stake: {f['central']:.2f}")
    ax.plot(xx, ym + f["reverse"] * (xx - xm), color=FIT_COL["reverse"], lw=1,
            label=f"reverse fit (not used): {f['reverse']:.2f}")
    ax.plot(xx, y[0] + f["anchored"] * (xx - xs5[0]), color=FIT_COL["anchored"], lw=1.5, ls=":",
            label=f"anchored at first pick (not used): {f['anchored']:.2f}")
    ax.plot(xx, ym + f["geometry"] * (xx - xm), color=FIT_COL["geometry"], lw=1.5, ls="--",
            label=f"geometry (homography): {f['geometry']:.2f}")
    ax.plot([xm], [ym], "+", color="k", ms=14, mew=2, label="centre of all picks")
    ax.set_xlabel("stake 5 image y (px), larger = surface lower")
    ax.set_ylabel(f"GCP{i} image y (px), larger = surface lower")
    ax.set_title(f"({'a' if i == 4 else 'b'}) GCP{i}: moved {y[-1] - y[0]:.0f} px while stake 5 moved {xs5[-1] - xs5[0]:.0f} px")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.3)


scatter_fits(fig.add_subplot(gs[0, 0]), 4)
scatter_fits(fig.add_subplot(gs[0, 1]), 2)

ax = fig.add_subplot(gs[0, 2])
xb = np.arange(4)
ax.bar(xb - 0.2, [fits[i]["ols"] for i in range(1, 5)], width=0.4, color=FIT_COL["ols"], label="current OLS")
ax.bar(xb + 0.2, [fits[i]["central"] for i in range(1, 5)], width=0.4, color=FIT_COL["central"],
       yerr=[[fits[i]["central"] - tls_tab.loc[i, "slope_p16"] for i in range(1, 5)],
             [tls_tab.loc[i, "slope_p84"] - fits[i]["central"] for i in range(1, 5)]],
       capsize=4, label="central ±1σ (bootstrap)")
ax.scatter(xb + 0.45, [fits[i]["geometry"] for i in range(1, 5)], marker="D", color=FIT_COL["geometry"], zorder=5,
           label="geometry (homography)")
ax.set_xticks(xb, [f"GCP{i}\n{scale[i]:.0f} px/m" for i in range(1, 5)])
ax.set_ylabel("px GCP movement per px stake movement")
ax.set_title(f"(c) slope per GCP (stake 5: {scale_stake:.0f} px/m; near GCPs look bigger)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

ax = fig.add_subplot(gs[1, 0:2])
jul = e4.loc[:pairs["timestamp"].max()]
ax.fill_between(jul.index, jul["y_p16"], jul["y_p84"], color=FIT_COL["central"], alpha=0.2, lw=0)
ax.plot(jul.index, jul["y_central"], color=FIT_COL["central"], lw=2, label="central model ±1σ")
ax.plot(jul.index, df_lin.set_index("gcp_id").loc[4, "intercept"] + fits[4]["ols"] * jul["y5"],
        color=FIT_COL["ols"], lw=2, label="current OLS model")
ax.plot(jul.index, y4[0] + fits[4]["anchored"] * (jul["y5"] - xs5[0]), color=FIT_COL["anchored"], lw=1.5, ls=":",
        label="anchored at first pick (not used)")
ax.plot(pairs["timestamp"], y4, linestyle="None", marker="o", markersize=7, markerfacecolor="none",
        markeredgecolor="red", label="GCP4 manual picks")
ax.set_ylabel("GCP4 image y (px), larger = surface lower")
ax.set_title("(d) GCP4 in July: the too-flat OLS line misses both ends of the picks")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[1, 2])
d_ols = fits[4]["ols"] * (e4["y5"] - e4["y5"].iloc[0])
d_cen = e4["y_central"] - e4["y_central"].iloc[0]
ax.fill_between(e4.index, e4["dy_p16"], e4["dy_p84"], color=FIT_COL["central"], alpha=0.2, lw=0)
ax.plot(e4.index, d_cen, color=FIT_COL["central"], lw=2,
        label=f"central: {d_cen.iloc[-1]:.0f} px ({e4['dy_p16'].iloc[-1]:.0f}-{e4['dy_p84'].iloc[-1]:.0f})")
ax.plot(e4.index, d_ols, color=FIT_COL["ols"], lw=2, label=f"current OLS: {d_ols.iloc[-1]:.0f} px")
ax.plot(pairs["timestamp"], y4 - y4[0], linestyle="None", marker="o", markersize=6, markerfacecolor="none",
        markeredgecolor="red", label="GCP4 manual picks")
sec = ax.secondary_yaxis("right", functions=(lambda v: v / pxm_cal, lambda v: v * pxm_cal))
sec.set_ylabel(f"m at {pxm_cal:.1f} px/m")
ax.set_ylabel("GCP4 movement since the first frame (px)")
ax.set_title("(e) GCP4 over the season")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(True, alpha=0.3)

fig.suptitle("GCP model: GCP image y = a + b · stake 5 image y, fitted on the July picks", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Spatio-temporal scatters of modelled data

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

# %% [markdown]
# ## PDH scatters of modelled data

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
)

df_kanu["PDH"] = (
    df_kanu["t_u"]
    .clip(lower=0)
)

df_kanu["cum_PDH"] = (
    df_kanu["t_u"]
    .clip(lower=0)
    .fillna(0)
    .cumsum()
)

# Aggregate hourly temperature data to one row per day
df_kanu_daily = (
    df_kanu
    .assign(datetime=df_kanu["datetime"].dt.normalize())
    .groupby("datetime", as_index=False)
    .agg(
        PDH=("PDH", "sum"),
        cum_PDH=("cum_PDH", "last"),
    )
)

print(df_kanu_daily.head())

# =====================================================
# MERGE MODELLED GCPS AND PDH DATA
# =====================================================

df_comb_mod = pd.merge(
    df_gcps_mod,
    df_kanu_daily,
    on="datetime",
    how="inner"
)

print("Merged rows:", len(df_comb_mod))
print(df_comb_mod.head())

# =====================================================
# SCATTER + REGRESSION
# =====================================================

x_var = "cum_PDH"
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

# %% [markdown]
# # 4. Modelled GCP img coordinates conversion for Hydro-STIV

# %%
mod_out_stiv = model_dir / "output_stiv"
Path(mod_out_stiv).mkdir(parents=True, exist_ok=True)

# %%
from pathlib import Path
import pandas as pd

# ==========================================================
# CONFIGURATION
# ==========================================================

INPUT_REAL = mod_in / "ilh-cam1-pt_gcps_real.csv"

INPUT_IMG_DIR = mod_out
OUTPUT_DIR = mod_out_stiv

OUTPUT_HEADERS = [
    "Name",
    "X(m)",
    "Y(m)",
    "Height(m)",
    "X(pixel)",
    "Y(pixel)"
]

HEIGHT_VALUE = 0

# Create output directory if necessary
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# READ REAL-WORLD COORDINATES
# ==========================================================

# No header in input file
real_df = pd.read_csv(INPUT_REAL, header=None)

# Expecting:
# column 0 -> X(m)
# column 1 -> Y(m)

# ==========================================================
# PROCESS ALL IMAGE FILES
# ==========================================================

for img_file in INPUT_IMG_DIR.glob("*_img_*.csv"):

    # Skip already processed files
    if "_img_stiv_" in img_file.name:
        continue

    # Read image coordinates (no header)
    img_df = pd.read_csv(img_file, header=None)

    # Sanity check
    if len(real_df) != len(img_df):
        print(
            f"Skipping {img_file.name}: "
            f"{len(real_df)} real points but "
            f"{len(img_df)} image points."
        )
        continue

    n_points = len(real_df)

    output_df = pd.DataFrame({
        "Name": [f"No.{i+1}" for i in range(n_points)],
        "X(m)": real_df.iloc[:, 0],
        "Y(m)": real_df.iloc[:, 1],
        "Height(m)": HEIGHT_VALUE,
        "X(pixel)": img_df.iloc[:, 0],
        "Y(pixel)": img_df.iloc[:, 1],
    })

    output_df = output_df[OUTPUT_HEADERS]

    output_name = img_file.name.replace(
        "_img_",
        "_img_stiv_"
    )

    output_file = OUTPUT_DIR / output_name

    output_df.to_csv(
        output_file,
        index=False
    )

    print(f"Created: {output_file.name}")

print("Finished.")

# %% [markdown]
# # 5. Surface lowering at the camera site: imagery vs air temperature, KAN_U and KAN_M
#
# Surface lowering since 2025-07-01 12:00 (the calibration image), from two independent sources:
# - **camera imagery**: GCP4 manual picks (July), stake 5 × 1.9 (band 1.8-2.0), the current GCP model
#   (`_modelling/output`) and the central model with its ±1σ band (section 3, `gcp_model_envelope.csv`),
#   all converted with the GCP4 emergence calibration (px per m). No temperature data used.
# - **air temperature**: camera-site logger (`hydro_ilh_Table1.dat`) PDH × the KAN_U surface lowering per °C·d
#   of each period (periods split at the August snowfalls; periods without melt at KAN_U are taken over from
#   KAN_U as they are). No camera imagery used.
#
# Stations: KAN_U surface (−z_surf_combined, 6 h mean; band: upper/lower boom), KAN_M surface (z_stake_cor)
# only while its spring snow lasts (midday of the first day with 12-18 UTC albedo < 0.6).
# PDH: hourly air temperature, gaps up to 3 h interpolated, clipped at 0, summed / 24 (°C·d).

# %%
# ==================================================
# SETTINGS AND DATA
# ==================================================
import re

cmp_t0 = pd.Timestamp("2025-07-01 12:00")     # start: calibration image (= first stake frame)
cmp_t1 = pd.Timestamp("2025-09-20 12:00")     # end of the season
cmp_tw = pd.Timestamp("2025-07-17 12:00")     # end of the snow-melt window
cmp_root = direc.parent.parent                # RIVeR/testing
cmp_edges = pd.to_datetime([
    "2025-07-01 12:00", "2025-07-17 12:00", "2025-07-24 12:00", "2025-08-05 12:00", "2025-08-14 12:00",
    "2025-08-20 12:00", "2025-08-26 12:00", "2025-09-05 12:00", "2025-09-20 12:00",
])                                            # KAN_U melt-rate periods, split at the August snowfalls
CMP_COL = {
    "camera site": "tab:green", "KAN_U": "k", "KAN_M": "#7b3294", "estimate": "#e08214",
    "current": "tab:blue", "central": "#e7298a", "picks": "tab:red",
}

# air temperature: camera-site logger (10-min values labelled at the interval end -> hourly means at the hour start)
logger = pd.read_csv(
    cmp_root / "hydro_ilh_Table1.dat",
    skiprows=[0, 2, 3],
    na_values=["NAN"],
    parse_dates=["TIMESTAMP"],
    index_col="TIMESTAMP"
)
t_cam = logger["AirTemp_C"].shift(-10, freq="min").resample("1h").mean()
st_ku = pd.read_csv(temp_dat, index_col=0, parse_dates=True)
st_km = pd.read_csv(cmp_root / "KAN_M_hour_new.csv", index_col=0, parse_dates=True, low_memory=False)


def pdh_cum(t, start, end):
    # cumulative PDH (°C·d); the value at time t is the PDH of the hours before t
    p = t.loc[start:end - pd.Timedelta("1h")].interpolate(limit=3).clip(lower=0).cumsum() / 24
    p.index = p.index + pd.Timedelta("1h")
    p.loc[start] = 0.0
    return p.sort_index()


sm6 = lambda s: s.rolling("6h", center=True).mean()
since = lambda s: s - s.loc[cmp_t0]

# station surface lowering since cmp_t0 (positive = surface lower)
low_ku = -since(sm6(st_ku["z_surf_combined"]))
low_ku_booms = [since(sm6(st_ku[c])) for c in ["z_boom_cor_l", "z_boom_cor_u"]]
alb_km = st_km["albedo"].loc[cmp_t0:cmp_t1].between_time("12:00", "18:00").resample("1D").mean()
km_snow_off = alb_km[alb_km < 0.6].index.min() + pd.Timedelta("12h")
low_km = since(sm6(st_km["z_stake_cor"])).loc[cmp_t0:km_snow_off]

# camera imagery -> metres with the GCP4 emergence calibration
cmp_calib = pd.read_csv(
    cam_dir / f"{cam}_emergence_calibration.csv",
    dtype={"point": str, "date": str, "time": str}
).iloc[0]
cmp_pxm = float(cmp_calib["px_per_m"])
ts_of = lambda f: pd.to_datetime("".join(re.search(r"(\d{8})_(\d{6})", f.name).groups()), format="%Y%m%d%H%M%S")


def img_y(folder, pattern, row):
    # image y of one point (row of the csv) per frame timestamp
    return pd.Series({ts_of(f): pd.read_csv(f, header=None).iat[row, 1] for f in folder.glob(pattern)}).sort_index().astype(float)


y_stake = img_y(cam_dir, f"{cam}_stake_point_*.csv", 0)
env4 = (
    pd.read_csv(cam_dir / "_modelling" / "gcp_model_envelope.csv", parse_dates=["timestamp"])
    .query("gcp_id == 4")
    .set_index("timestamp")
    .sort_index()
)
cam_low = {
    "GCP4 manual picks": since(img_y(cam_dir, f"{cam}_gcps_img_*.csv", 3)) / cmp_pxm,
    "GCP4 current model": since(img_y(cam_dir / "_modelling" / "output", f"{cam}_gcps_img_*.csv", 3)) / cmp_pxm,
    "GCP4 central model": since(env4["y_central"]) / cmp_pxm,
    "stake 5 × 1.9": since(y_stake) * 1.9 / cmp_pxm,
}
cam_band = {
    # dy_* in the envelope file are relative to the first stake frame (= cmp_t0)
    "GCP4 central model": (env4["dy_p16"] / cmp_pxm, env4["dy_p84"] / cmp_pxm),
    "stake 5 × 1.9": (since(y_stake) * 1.8 / cmp_pxm, since(y_stake) * 2.0 / cmp_pxm),
}

print(f"Calibration: {cmp_calib['point']} emergence {cmp_calib['emergence_m']} m on {cmp_calib['date']} -> {cmp_pxm:.1f} px/m")
print(f"KAN_M spring snow gone: {km_snow_off}")

# %%
# ==================================================
# SURFACE LOWERING FROM AIR TEMPERATURE (camera-site PDH × KAN_U lowering per °C·d of each period)
# ==================================================
pdh_cam = pdh_cum(t_cam, cmp_t0, cmp_t1)
pdh_ku = pdh_cum(st_ku["t_u"], cmp_t0, cmp_t1)


def from_temperature(L_u):
    # camera-site lowering hour by hour from a KAN_U lowering record: in each period KAN_U's lowering per °C·d
    # times the camera-site PDH; periods without melt at KAN_U (snowfall, settling) are taken over as they are
    out = pd.Series(np.nan, index=pdh_cam.index)
    level = 0.0
    for a, b in zip(cmp_edges[:-1], cmp_edges[1:]):
        seg = pdh_cam.loc[a:b].index
        d_pdh = pdh_ku.loc[b] - pdh_ku.loc[a]
        if d_pdh > 0.3:
            out.loc[seg] = level + (L_u.loc[b] - L_u.loc[a]) / d_pdh * (pdh_cam.loc[seg] - pdh_cam.loc[a])
        else:
            out.loc[seg] = level + (L_u.reindex(seg) - L_u.loc[a]).values
        level = out.loc[b]
    return out


low_est = from_temperature(low_ku)
est_booms = [from_temperature(L) for L in low_ku_booms]
low_est_lo = np.fmin(np.fmin(*est_booms), low_est)
low_est_hi = np.fmax(np.fmax(*est_booms), low_est)

period_rows = []
for a, b in zip(cmp_edges[:-1], cmp_edges[1:]):
    d_pdh = pdh_ku.loc[b] - pdh_ku.loc[a]
    d_low = low_ku.loc[b] - low_ku.loc[a]
    period_rows.append({
        "period": f"{a:%m-%d} -> {b:%m-%d}",
        "KAN_U_lowering_m": d_low,
        "KAN_U_PDH": d_pdh,
        "KAN_U_mm_per_degC_d": 1000 * d_low / d_pdh if d_pdh > 0.3 else np.nan,
        "camera_PDH": pdh_cam.loc[b] - pdh_cam.loc[a],
        "camera_lowering_m": low_est.loc[b] - low_est.loc[a],
    })

print(f"PDH {cmp_t0:%m-%d} -> {cmp_t1:%m-%d}: camera site {pdh_cam.iloc[-1]:.1f}, KAN_U {pdh_ku.iloc[-1]:.1f} °C·d")
pd.DataFrame(period_rows).set_index("period").round(3)

# %%
# ==================================================
# PLOT: surface lowering over the season
# ==================================================
seas = lambda s: s.loc[cmp_t0:cmp_t1]

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(seas(low_ku).index, seas(low_ku), color=CMP_COL["KAN_U"], lw=1.5, label="KAN_U surface (1848 m)")
ax.plot(low_km.index, low_km, color=CMP_COL["KAN_M"], lw=1.5,
        label=f"KAN_M surface (1265 m), until spring snow gone {km_snow_off:%m-%d}")
ax.fill_between(low_est.index, low_est_lo, low_est_hi, color=CMP_COL["estimate"], alpha=0.15, lw=0)
ax.plot(low_est.index, low_est, color=CMP_COL["estimate"], lw=1.8,
        label="camera site: from its air temperature (KAN_U rate per period × camera PDH; band: KAN_U booms)")

for name, color, label in [
    ("stake 5 × 1.9", CMP_COL["camera site"], "camera site: stake 5 × 1.9 (band 1.8-2.0)"),
    ("GCP4 central model", CMP_COL["central"], "camera site: GCP4 central model (band ±1σ)"),
]:
    b_lo, b_hi = (seas(b) for b in cam_band[name])
    ax.fill_between(b_lo.index, b_lo, b_hi, color=color, alpha=0.2, lw=0)
    ax.plot(seas(cam_low[name]).index, seas(cam_low[name]), color=color, lw=1.5, label=label)

ser = seas(cam_low["GCP4 current model"])
ax.plot(ser.index, ser, color=CMP_COL["current"], lw=1.2, label="camera site: GCP4 current model (_modelling/output)")
ser = seas(cam_low["GCP4 manual picks"])
ax.plot(ser.index, ser, linestyle="None", marker="o", markersize=6, markerfacecolor="none",
        markeredgecolor=CMP_COL["picks"], label="camera site: GCP4 manual picks")

ax.axhline(0, color="k", lw=0.6)
ax.set_xlim(cmp_t0 - pd.Timedelta("1D"), cmp_t1 + pd.Timedelta("1D"))
ax.set_ylabel(f"Surface lowering since {cmp_t0:%Y-%m-%d %H:%M} (m)")
ax.set_title(f"Camera site vs PROMICE stations | camera calibration {cmp_pxm:.1f} px/m "
             f"({cmp_calib['point']} emergence {cmp_calib['emergence_m']} m on {cmp_calib['date']})")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=8.5)

plt.tight_layout()
plt.show()

# %%
# ==================================================
# SNOW-MELT WINDOW cmp_t0 -> cmp_tw: temperatures and PDH at the three sites
# ==================================================
CMP_SITES = ["KAN_M", "camera site", "KAN_U"]
SITE_COL = {"camera site": CMP_COL["camera site"], "KAN_U": CMP_COL["KAN_U"], "KAN_M": CMP_COL["KAN_M"]}
cmp_hours = pd.date_range(cmp_t0, cmp_tw - pd.Timedelta("1h"), freq="1h")

Tw = pd.DataFrame({
    "camera site": t_cam.reindex(cmp_hours),
    "KAN_U": st_ku["t_u"].reindex(cmp_hours),
    "KAN_M": st_km["t_u"].reindex(cmp_hours),
}).interpolate(limit=3)

pdh_w = Tw.clip(lower=0).cumsum() / 24
pdh_w.index = pdh_w.index + pd.Timedelta("1h")
pdh_w.loc[cmp_t0] = 0.0
pdh_w = pdh_w.sort_index()

t_metrics = pd.DataFrame({
    site: {
        "T_mean": Tw[site].mean(),
        "T_min": Tw[site].min(),
        "T_max": Tw[site].max(),
        "hours_above_0_pct": 100 * (Tw[site] > 0).mean(),
        "PDH_degC_d": pdh_w[site].iloc[-1],
        "r_hourly_vs_camera": Tw[site].corr(Tw["camera site"]),
        "dT_vs_camera": (Tw[site] - Tw["camera site"]).mean(),
    }
    for site in ["camera site", "KAN_U", "KAN_M"]
}).T

# KAN_U lowering per °C·d in the window (= first period of the estimate above), booms as range
ddf_w = low_ku.loc[cmp_tw] / pdh_w["KAN_U"].iloc[-1]
ddf_w_lo, ddf_w_hi = sorted(L.loc[cmp_tw] / pdh_w["KAN_U"].iloc[-1] for L in low_ku_booms)
t_metrics.round(2)

# %%
# ==================================================
# PLOT: snow-melt window
# ==================================================
ww = lambda s: s.loc[cmp_t0:cmp_tw]

fig, axes = plt.subplots(2, 3, figsize=(19, 10))

ax = axes[0, 0]
for site in CMP_SITES:
    ax.plot(Tw.index, Tw[site], color=SITE_COL[site], lw=1,
            label=f"{site} (mean {t_metrics.loc[site, 'T_mean']:+.2f} °C)")
ax.axhline(0, color="0.4", lw=0.8, ls=":")
ax.set_ylabel("Air temperature (°C)")
ax.set_title("(a) hourly air temperature")

ax = axes[0, 1]
for site in ["KAN_M", "KAN_U"]:
    ax.scatter(Tw["camera site"], Tw[site], s=6, color=SITE_COL[site], alpha=0.5,
               label=f"{site}: r {t_metrics.loc[site, 'r_hourly_vs_camera']:.2f}, "
                     f"Δ {t_metrics.loc[site, 'dT_vs_camera']:+.2f} °C")
t_lim = [Tw.min().min() - 0.5, Tw.max().max() + 0.5]
ax.plot(t_lim, t_lim, color="0.4", lw=0.8, ls="--", label="1:1")
ax.set_xlim(t_lim)
ax.set_ylim(t_lim)
ax.set_xlabel("Camera-site air temperature (°C)")
ax.set_ylabel("Station air temperature (°C)")
ax.set_title("(b) hourly temperature vs camera site")

ax = axes[0, 2]
for site in ["KAN_M", "KAN_U"]:
    dd = (Tw[site] - Tw["camera site"]).groupby(Tw.index.hour)
    hc = dd.mean().index + 0.5        # hourly means labelled at the hour start -> plot at the hour centre
    ax.plot(hc, dd.mean(), color=SITE_COL[site], label=f"{site} − camera site")
    ax.fill_between(hc, dd.quantile(0.25), dd.quantile(0.75), color=SITE_COL[site], alpha=0.15, lw=0)
ax.axhline(0, color="0.4", lw=0.8)
ax.axvline(15.2, color="tab:orange", lw=1, ls="--", label="local solar noon (15:12 UTC)")
ax.set_xticks(range(0, 24, 3))
ax.set_xlabel("Time of day (UTC)")
ax.set_ylabel("Temperature difference (°C)")
ax.set_title("(c) diurnal temperature difference (mean, IQR)")

ax = axes[1, 0]
for site in CMP_SITES:
    ax.plot(pdh_w.index, pdh_w[site], color=SITE_COL[site], lw=1.5, label=f"{site}: {pdh_w[site].iloc[-1]:.1f} °C·d")
ax.set_ylabel(f"Cumulative PDH since {cmp_t0:%m-%d %H:%M} (°C·d)")
ax.set_title("(d) cumulative positive degree-hours")

ax = axes[1, 1]
ax.plot(ww(low_km).index, ww(low_km), color=CMP_COL["KAN_M"], lw=1.5, label="KAN_M surface (stake sonic), until spring snow gone")
ax.fill_between(ww(low_ku_booms[0]).index, ww(low_ku_booms[0]), ww(low_ku_booms[1]), color=CMP_COL["KAN_U"], alpha=0.15, lw=0)
ax.plot(ww(low_ku).index, ww(low_ku), color=CMP_COL["KAN_U"], lw=1.5, label="KAN_U surface (band: upper/lower boom)")
ax.fill_between(ww(low_est).index, ww(low_est_lo), ww(low_est_hi), color=CMP_COL["estimate"], alpha=0.15, lw=0)
ax.plot(ww(low_est).index, ww(low_est), color=CMP_COL["estimate"], lw=1.8,
        label=f"camera site: from its air temperature ({1000 * ddf_w:.1f} mm per °C·d × PDH)")
for name, color in [("stake 5 × 1.9", CMP_COL["camera site"]), ("GCP4 central model", CMP_COL["central"])]:
    b_lo, b_hi = (ww(b) for b in cam_band[name])
    ax.fill_between(b_lo.index, b_lo, b_hi, color=color, alpha=0.2, lw=0)
    ax.plot(ww(cam_low[name]).index, ww(cam_low[name]), color=color, lw=1.5, label=f"camera site: {name}")
ser = ww(cam_low["GCP4 current model"])
ax.plot(ser.index, ser, color=CMP_COL["current"], lw=1.2, label="camera site: GCP4 current model")
ser = ww(cam_low["GCP4 manual picks"])
ax.plot(ser.index, ser, linestyle="None", marker="o", markersize=6, markerfacecolor="none",
        markeredgecolor=CMP_COL["picks"], label="camera site: GCP4 manual picks")
ax.axvline(km_snow_off, color=CMP_COL["KAN_M"], lw=0.8, ls="--")
ax.axhline(0, color="0.4", lw=0.8)
ax.set_ylabel(f"Surface lowering since {cmp_t0:%m-%d %H:%M} (m)")
ax.set_title("(e) surface lowering")

ax = axes[1, 2]
cmp_days = pd.date_range(cmp_t0, cmp_tw, freq="1D")
for name, ser, pdh, color, kw in [
    ("KAN_M", ww(low_km), pdh_w["KAN_M"], CMP_COL["KAN_M"], {}),
    ("KAN_U", ww(low_ku), pdh_w["KAN_U"], CMP_COL["KAN_U"], {}),
    ("stake 5 × 1.9", ww(cam_low["stake 5 × 1.9"]), pdh_w["camera site"], CMP_COL["camera site"], {}),
    ("GCP4 current model", ww(cam_low["GCP4 current model"]), pdh_w["camera site"], CMP_COL["current"], {"marker": "s"}),
    ("GCP4 central model", ww(cam_low["GCP4 central model"]), pdh_w["camera site"], CMP_COL["central"], {"marker": "D"}),
    ("GCP4 manual picks", ww(cam_low["GCP4 manual picks"]), pdh_w["camera site"], CMP_COL["picks"], {"markerfacecolor": "none"}),
]:
    yv = ser.reindex(cmp_days, method="nearest", tolerance=pd.Timedelta("1h"))
    xv = pdh.reindex(cmp_days)
    ok = yv.notna() & xv.notna()
    ax.plot(xv[ok], yv[ok], **{"marker": "o", "linestyle": "None", "markersize": 5, "color": color, **kw})
    fit = linregress(xv[ok], yv[ok])
    xx = np.array([0, xv[ok].max()])
    ax.plot(xx, fit.intercept + fit.slope * xx, color=color, lw=1, ls="--",
            label=f"{name}: {1000 * fit.slope:.1f} mm per °C·d (r {fit.rvalue:.2f}, "
                  f"{'p < 0.01' if fit.pvalue < 0.01 else f'p {fit.pvalue:.2f}'})")
xx = np.array([0, pdh_w["camera site"].iloc[-1]])
ax.fill_between(xx, ddf_w_lo * xx, ddf_w_hi * xx, color=CMP_COL["estimate"], alpha=0.12, lw=0)
ax.plot(xx, ddf_w * xx, color=CMP_COL["estimate"], lw=1.8,
        label=f"camera site, from air temperature: {1000 * ddf_w:.1f} mm per °C·d (band: KAN_U booms)")
ax.axhline(0, color="0.4", lw=0.8)
ax.set_xlabel("Cumulative PDH of the same site (°C·d)")
ax.set_ylabel("Surface lowering (m)")
ax.set_title("(f) lowering vs own PDH (daily 12:00 samples)")

for ax in axes.flat:
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
for ax in [axes[0, 0], axes[1, 0], axes[1, 1]]:
    ax.set_xlim(cmp_t0, cmp_tw)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
fig.suptitle(f"{cmp_t0:%m-%d %H:%M} → {cmp_tw:%m-%d %H:%M}: KAN_M (1265 m), camera site (~1590 m), KAN_U (1848 m) | "
             f"camera calibration {cmp_pxm:.1f} px/m", fontsize=12)

plt.tight_layout()
plt.show()

# %%
# ==================================================
# PLOT: camera site, from air temperature vs from imagery (snow-melt window)
# ==================================================
cmp_obs = [
    ("GCP4 manual picks", CMP_COL["picks"], "o", {"markerfacecolor": "none"}),
    ("GCP4 current model", CMP_COL["current"], "s", {}),
    ("GCP4 central model", CMP_COL["central"], "D", {}),
    ("stake 5 × 1.9", CMP_COL["camera site"], "o", {}),
]

fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.6, 1]})

ax = axes[0]
ax.fill_between(ww(low_est).index, ww(low_est_lo), ww(low_est_hi), color=CMP_COL["estimate"], alpha=0.15, lw=0)
ax.plot(ww(low_est).index, ww(low_est), color=CMP_COL["estimate"], lw=1.8,
        label=f"from air temperature: {low_est.loc[cmp_tw]:.2f} m")
for name, color, mk, kw in cmp_obs:
    if name in cam_band:
        b_lo, b_hi = (ww(b) for b in cam_band[name])
        ax.fill_between(b_lo.index, b_lo, b_hi, color=color, alpha=0.2, lw=0)
    ser = ww(cam_low[name]).dropna()
    if name == "GCP4 manual picks":
        ax.plot(ser.index, ser, linestyle="None", marker=mk, markersize=6, color=color, **kw,
                label=f"{name}: {ser.iloc[-1]:.2f} m")
    else:
        ax.plot(ser.index, ser, color=color, lw=1.4, label=f"{name}: {ser.iloc[-1]:.2f} m")
ax.axhline(0, color="0.4", lw=0.8)
ax.set_xlim(cmp_t0, cmp_tw)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.set_ylabel(f"Surface lowering since {cmp_t0:%m-%d %H:%M} (m)")
ax.set_title(f"(a) camera site from air temperature vs from imagery ({cmp_pxm:.1f} px/m)")

ax = axes[1]
for name, color, mk, kw in cmp_obs:
    ser = ww(cam_low[name]).dropna()
    ser = ser[ser.index.hour == 12]                 # daily 12:00 frames
    ev = low_est.reindex(ser.index, method="nearest", tolerance=pd.Timedelta("1h"))
    dv = ev - ser
    ax.plot(ev, ser, linestyle="None", marker=mk, markersize=6, color=color, **kw,
            label=f"{name}: r {ev.corr(ser):.2f}, bias {100 * dv.mean():+.1f} cm, RMSE {100 * np.sqrt((dv ** 2).mean()):.1f} cm")
xx = np.array([0, low_est.loc[cmp_tw]])
ax.fill_between(xx, ddf_w_lo / ddf_w * xx, ddf_w_hi / ddf_w * xx, color=CMP_COL["estimate"], alpha=0.12, lw=0)
ax.plot(xx, xx, color=CMP_COL["estimate"], lw=1.8, label="from air temperature (band: KAN_U booms)")
t_lim = [min(-0.1, min(ww(cam_low[n]).min() for n, *_ in cmp_obs) - 0.01),
         max(ww(cam_low[n]).max() for n, *_ in cmp_obs) + 0.02]
ax.plot(t_lim, t_lim, color="0.4", lw=0.8, ls="--", label="1:1")
ax.set_xlim(t_lim)
ax.set_ylim(t_lim)
ax.set_xlabel("Lowering from camera-site air temperature (m)")
ax.set_ylabel("Lowering from camera imagery (m)")
ax.set_title("(b) daily 12:00 frames; bias = from temperature − from imagery")

for ax in axes:
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left")

plt.tight_layout()
plt.show()

# %%
# ==================================================
# METRICS: camera-site series vs the air-temperature estimate (daily 12:00 frames)
# ==================================================
cmp_rows = []
for name in ["GCP4 central model", "GCP4 current model", "stake 5 × 1.9", "GCP4 manual picks"]:
    ser = seas(cam_low[name]).dropna()
    s12 = ser[ser.index.hour == 12]
    e12 = low_est.reindex(s12.index, method="nearest", tolerance=pd.Timedelta("1h"))
    dv = s12 - e12
    cmp_rows.append({
        "series": name,
        "until": f"{ser.index.max():%m-%d}",
        "lowering_m": ser.iloc[-1],
        "from_temperature_m": low_est.loc[ser.index.max()],
        "mean_diff_cm": 100 * dv.mean(),           # imagery − temperature
        "rmse_cm": 100 * np.sqrt((dv ** 2).mean()),
        "n": len(dv),
    })

b_lo, b_hi = (seas(b) for b in cam_band["GCP4 central model"])
e_band = low_est.reindex(b_lo.index, method="nearest", tolerance=pd.Timedelta("1h"))
print(f"Central model ±1σ at {cmp_t1:%m-%d}: {b_lo.loc[cmp_t1]:.3f} - {b_hi.loc[cmp_t1]:.3f} m; "
      f"air-temperature estimate inside the band on {100 * ((e_band >= b_lo) & (e_band <= b_hi)).mean():.0f} % of the frames")
pd.DataFrame(cmp_rows).set_index("series").round(3)

# %%
# ==================================================
# PLOT: why the air-temperature estimate uses periods (documentation)
# ==================================================
# One fixed factor (KAN_U's lowering per °C·d in the snow-melt window) applied to KAN_U's own temperature
# undershoots KAN_U in July-August: the factor rose when the snow darkened (albedo) and fresh snow settled,
# and August snowfall is not melt. The estimate above therefore uses KAN_U's factor of each period.
fixed_ku = ddf_w * pdh_ku
fixed_cam = ddf_w * pdh_cam
per = pd.DataFrame(period_rows)
mid = cmp_edges[:-1] + (cmp_edges[1:] - cmp_edges[:-1]) / 2
width = (cmp_edges[1:] - cmp_edges[:-1]) * 0.85

fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [1.6, 1]}, sharex=True)

ax = axes[0]
ax.plot(seas(low_ku).index, seas(low_ku), color=CMP_COL["KAN_U"], lw=1.5, label="KAN_U observed")
ax.plot(fixed_ku.index, fixed_ku, color=CMP_COL["KAN_U"], lw=1.5, ls="--",
        label=f"KAN_U from its own temperature, fixed {1000 * ddf_w:.1f} mm per °C·d ({cmp_t0:%m-%d} → {cmp_tw:%m-%d})")
ax.plot(fixed_cam.index, fixed_cam, color=CMP_COL["estimate"], lw=1.6, ls=":",
        label="camera site from its temperature, same fixed factor")
ax.plot(low_est.index, low_est, color=CMP_COL["estimate"], lw=1.8,
        label="camera site from its temperature, KAN_U factor per period (used above)")
ser = seas(cam_low["stake 5 × 1.9"])
ax.plot(ser.index, ser, color=CMP_COL["camera site"], lw=1, alpha=0.6, label="camera site observed: stake 5 × 1.9")
for e in cmp_edges[1:-1]:
    ax.axvline(e, color="0.7", lw=0.6, ls=":")
ax.axhline(0, color="0.4", lw=0.8)
ax.set_ylabel(f"Surface lowering since {cmp_t0:%m-%d %H:%M} (m)")
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.bar(mid, per["KAN_U_mm_per_degC_d"], width=width, color="0.75", edgecolor="k", lw=0.6,
       label="KAN_U lowering per °C·d of its own PDH")
ax.axhline(1000 * ddf_w, color=CMP_COL["estimate"], lw=1.5, ls=":", label=f"fixed factor ({1000 * ddf_w:.1f})")
for m_, v, d_pdh, d_low in zip(mid, per["KAN_U_mm_per_degC_d"], per["KAN_U_PDH"], per["KAN_U_lowering_m"]):
    if np.isnan(v):
        what = f"snowfall\n{100 * d_low:+.0f} cm" if d_low < 0 else f"no melt\n{100 * d_low:+.0f} cm"
        ax.text(m_, 1, f"{what}\nPDH {d_pdh:.1f}", ha="center", va="bottom", fontsize=7.5)
ax.set_ylabel("mm per °C·d")
ax.grid(True, alpha=0.3, axis="y")
ax2 = ax.twinx()
alb_d = st_ku["albedo"].loc[cmp_t0:cmp_t1].between_time("12:00", "18:00").resample("1D").mean()
ax2.plot(alb_d.index, alb_d, color="tab:blue", lw=1.3, label="KAN_U albedo (12-18 UTC)")
ax2.set_ylabel("Albedo", color="tab:blue")
ax2.tick_params(axis="y", colors="tab:blue")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper right")
ax.set_xlim(cmp_t0 - pd.Timedelta("1D"), cmp_t1 + pd.Timedelta("1D"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

fig.suptitle("Fixed degree-day factor vs KAN_U's own lowering per degree-day (documentation)")
plt.tight_layout()
plt.show()

# %%
