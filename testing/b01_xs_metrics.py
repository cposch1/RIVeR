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
from river.config import bathy_dir, pts_dir
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

# ==================================================
# USER SETTINGS
# ==================================================

cam = "ilh-cam1-pt"

# old XS data for now; switch back to bathy_dir once the new XS data is done
xs_dir = bathy_dir# / "_old_data"

pt_files = [
    #"Stage01-data-2026-04-24 10_36_57.csv",
    #"Stage02-data-2026-04-24 10_40_48.csv",
]

start_date = pd.to_datetime("2025-07-14 00:00")
end_date   = pd.to_datetime("2025-09-01 00:00")

# --------------------------------------------------
# choose reference for time window:
# "xs" = align to XS coverage
# "pt" = align to PT coverage
# --------------------------------------------------
time_reference = "xs"   # or "pt"

# %%
# ==================================================
# LOAD XS DATA
# ==================================================

cam_dir = xs_dir / cam
records = []

for f in sorted(cam_dir.glob(f"{cam}_xs_coord_*.csv")):

    try:
        timestamp = f.stem.replace(f"{cam}_xs_coord_", "")
        dt = pd.to_datetime(timestamp, format="%Y%m%d_%H%M%S")

        df_tmp = pd.read_csv(f, header=None)
        print(df_tmp)

        # distance between XS end points, works for relative (local)
        # and absolute (UTM) coordinates
        xs_length = float(np.hypot(
            df_tmp.iloc[1, 0] - df_tmp.iloc[0, 0],
            df_tmp.iloc[1, 1] - df_tmp.iloc[0, 1]
        ))

        records.append({
            "datetime": dt,
            "xs_length": xs_length
        })

    except Exception as e:
        print(f"Skipping {f.name}: {e}")

df_xs = (
    pd.DataFrame(records)
    .dropna()
    .sort_values("datetime")
    .reset_index(drop=True)
)

# ==================================================
# LOAD ALL PT DATA FIRST (for optional time reference)
# ==================================================

pt_data = []

for pt_file in pt_files:

    pt_path = pts_dir / pt_file

    if not pt_path.exists():
        print("Missing:", pt_file)
        continue

    df_pt = pd.read_csv(pt_path)

    depth_col = [c for c in df_pt.columns if "Water_deep" in c]
    if not depth_col:
        continue
    depth_col = depth_col[0]

    df_pt["Time"] = pd.to_datetime(df_pt["Time"], errors="coerce")
    df_pt = df_pt.dropna(subset=["Time"])

    df_pt[depth_col] = pd.to_numeric(df_pt[depth_col], errors="coerce")

    df_pt = df_pt.dropna(subset=[depth_col])

    pt_data.append((pt_file, df_pt, depth_col))

# ==================================================
# TIME WINDOW SELECTION
# ==================================================

if time_reference == "xs":
    xmin = df_xs["datetime"].min()
    xmax = df_xs["datetime"].max()
else:
    all_pt_times = pd.concat([df[1]["Time"] for df in pt_data])
    xmin = all_pt_times.min()
    xmax = all_pt_times.max()

print("\nTime window:", xmin, "→", xmax)

# ==================================================
# PLOT SETUP
# ==================================================

fig, ax1 = plt.subplots(figsize=(14, 6))

# Make background transparent
fig.patch.set_alpha(0)
ax1.set_facecolor("none")

# 1. invisible line
ax1.plot(
    df_xs["datetime"],
    df_xs["xs_length"],
    "-",
    color="white",
    linewidth=2.5,
    alpha=0,   # hides line
)

# 2. visible markers
ax1.plot(
    df_xs["datetime"],
    df_xs["xs_length"],
    linestyle="None",
    marker="o",
    markerfacecolor="white",
    markeredgecolor="white",
    label="Channel width"
)

# Axis styling (white)
ax1.set_xlabel("Date", color="white")
ax1.set_ylabel("Channel width (m)", color="white")

ax1.set_ylim(0,25)
ax1.tick_params(colors="white")
for spine in ax1.spines.values():
    spine.set_color("white")

ax1.grid(True, color="white", alpha=0.2)




ax1.set_xlim(start_date, end_date)


# Get full axis range (matplotlib stores dates as numbers)
xmin_plot, xmax_plot = ax1.get_xlim()

# Convert to datetime
start = pd.to_datetime(mdates.num2date(xmin_plot)).normalize()
end = pd.to_datetime(mdates.num2date(xmax_plot)).normalize()

# Create daily ticks at 00:00 across FULL plot range
daily_ticks = pd.date_range(start=start, end=end, freq="1D")

ax1.set_xticks(daily_ticks)

# Format labels
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

# Rotate labels
plt.setp(ax1.get_xticklabels(), rotation=90, ha="center", color="white")


# Secondary axis
ax2 = ax1.twinx()
ax2.set_facecolor("none")

colors = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange"]

# ==================================================
# PLOT PT DATA (DASHED)
# ==================================================

for i, (pt_file, df_pt, depth_col) in enumerate(pt_data):

    df_pt = df_pt.set_index("Time").sort_index()

    df_pt_2h = (
        df_pt
        .resample("2h", origin="start_day", label="left", closed="left")
        .mean()
        .reset_index()
    )

    df_pt_2h = df_pt_2h[
        (df_pt_2h["Time"] >= xmin) &
        (df_pt_2h["Time"] <= xmax)
    ]

    if df_pt_2h.empty:
        continue

    # PT = dashed line
    ax2.plot(
        df_pt_2h["Time"],
        df_pt_2h[depth_col],
        ":",
        linewidth=1,
        #color=colors[i % len(colors)],
        color="white",
        label=pt_file
    )

# Axis styling (white)
ax2.set_ylim(168,178)
ax2.set_ylabel("Water depth (cm)", color="white")
ax2.tick_params(colors="white")
for spine in ax2.spines.values():
    spine.set_color("white")

# Legends (white text)
leg1 = ax1.legend(loc="upper left")
leg2 = ax2.legend(loc="upper right")

for text in leg1.get_texts():
    text.set_color("white")
for text in leg2.get_texts():
    text.set_color("white")


leg1.get_frame().set_facecolor("none")
leg2.get_frame().set_facecolor("none")

leg1.get_frame().set_alpha(0.5)
leg2.get_frame().set_alpha(0.5)


# Title
#plt.title(f"{cam}: Channel width and water depth", color="white")
plt.tight_layout()

plt.savefig("channel-width_pt_2.svg")
plt.show()

# %%
df_xs.to_csv("xs_lengths.csv", index=False)

# %%
from river.config import bathy_dir, pts_dir
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

# ==================================================
# USER SETTINGS
# ==================================================

cam = "ilh-cam1-pt"
kan_u = pd.read_csv('KAN_U_day_new.csv', index_col=0, parse_dates=True)

# old XS data for now; switch back to bathy_dir once the new XS data is done
xs_dir = bathy_dir# / "_old_data"

start_date = pd.to_datetime("2025-07-14 00:00")
end_date   = pd.to_datetime("2025-09-01 00:00")

# --------------------------------------------------
# choose reference for time window:
# "xs" = align to XS coverage
# "pt" = align to PT coverage
# --------------------------------------------------
time_reference = "xs"   # or "pt"

# ==================================================
# LOAD XS DATA
# ==================================================

cam_dir = xs_dir / cam
records = []

for f in sorted(cam_dir.glob(f"{cam}_xs_coord_*.csv")):

    try:
        timestamp = f.stem.replace(f"{cam}_xs_coord_", "")
        dt = pd.to_datetime(timestamp, format="%Y%m%d_%H%M%S")

        df_tmp = pd.read_csv(f, header=None)
        print(df_tmp)

        # distance between XS end points, works for relative (local)
        # and absolute (UTM) coordinates
        xs_length = float(np.hypot(
            df_tmp.iloc[1, 0] - df_tmp.iloc[0, 0],
            df_tmp.iloc[1, 1] - df_tmp.iloc[0, 1]
        ))

        records.append({
            "datetime": dt,
            "xs_length": xs_length
        })

    except Exception as e:
        print(f"Skipping {f.name}: {e}")

df_xs = (
    pd.DataFrame(records)
    .dropna()
    .sort_values("datetime")
    .reset_index(drop=True)
)

# ==================================================
# LOAD TEMP data
# ==================================================

yr = 2025
years = np.arange(yr, yr+1)

# ==================================================
# TIME WINDOW SELECTION
# ==================================================

if time_reference == "xs":
    xmin = df_xs["datetime"].min()
    xmax = df_xs["datetime"].max()
else:
    all_pt_times = pd.concat([df[1]["Time"] for df in pt_data])
    xmin = all_pt_times.min()
    xmax = all_pt_times.max()

print("\nTime window:", xmin, "→", xmax)

# ==================================================
# PLOT SETUP
# ==================================================

fig, ax1 = plt.subplots(figsize=(14, 6))

# Make background transparent
fig.patch.set_alpha(0)
ax1.set_facecolor("none")

# 1. invisible line
ax1.plot(
    df_xs["datetime"],
    df_xs["xs_length"],
    "-",
    color="white",
    linewidth=2.5,
    alpha=0,   # hides line
)

# 2. visible markers
ax1.plot(
    df_xs["datetime"],
    df_xs["xs_length"],
    linestyle="None",
    marker="o",
    markerfacecolor="white",
    markeredgecolor="white",
    label="Channel width"
)

# Axis styling (white)
ax1.set_xlabel("Date", color="white")
ax1.set_ylabel("Channel width (m)", color="white")

ax1.set_ylim(0,25)
ax1.tick_params(colors="white")
for spine in ax1.spines.values():
    spine.set_color("white")

ax1.grid(True, color="white", alpha=0.2)




ax1.set_xlim(start_date, end_date)


# Get full axis range (matplotlib stores dates as numbers)
xmin_plot, xmax_plot = ax1.get_xlim()

# Convert to datetime
start = pd.to_datetime(mdates.num2date(xmin_plot)).normalize()
end = pd.to_datetime(mdates.num2date(xmax_plot)).normalize()

# Create daily ticks at 00:00 across FULL plot range
daily_ticks = pd.date_range(start=start, end=end, freq="1D")

ax1.set_xticks(daily_ticks)

# Format labels
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

# Rotate labels
plt.setp(ax1.get_xticklabels(), rotation=90, ha="center", color="white")


# Secondary axis
ax2 = ax1.twinx()
ax2.set_facecolor("none")

colors = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange"]

# ==================================================
# PLOT TEMP DATA (DASHED)
# ==================================================


for y in years:
    data = kan_u.loc[f"{y}-07-14":f"{y}-09-01"].t_u

    
    ax2.plot(
        data.index,
        data.values,
        linestyle="--",
        linewidth=1,
        color="white",
        label="KAN_U 2m air temperature"
    )


ax2.set_ylim(-15,10)

# Axis styling (white)
#ax2.set_ylim(168,178)
ax2.set_ylabel("Air temperature(°C)", color="white")
ax2.tick_params(colors="white")
for spine in ax2.spines.values():
    spine.set_color("white")

# Legends (white text)
leg1 = ax1.legend(loc="upper left")
leg2 = ax2.legend(loc="upper right")

for text in leg1.get_texts():
    text.set_color("white")
for text in leg2.get_texts():
    text.set_color("white")


leg1.get_frame().set_facecolor("none")
leg2.get_frame().set_facecolor("none")

leg1.get_frame().set_alpha(0.5)
leg2.get_frame().set_alpha(0.5)


# Title
#plt.title(f"{cam}: Channel width and water depth", color="white")
plt.tight_layout()

plt.savefig("channel-width_temp.svg")
plt.show()

# %%
from river.config import pts_dir
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==================================================
# USER SETTINGS
# ==================================================

pt_file = "Stage02-data-2026-04-24 10_40_48.csv"

start_date = pd.to_datetime("2025-05-10 00:00")
end_date   = pd.to_datetime("2025-07-17 00:00")

# ==================================================
# LOAD DATA
# ==================================================

# --- PT data ---
pt_path = pts_dir / pt_file
df_pt = pd.read_csv(pt_path)

depth_col = [c for c in df_pt.columns if "Water_deep" in c][0]

df_pt["Time"] = pd.to_datetime(df_pt["Time"])
df_pt = df_pt.set_index("Time").sort_index()

# ✅ HOURLY RESAMPLING
df_pt_1h = df_pt.resample("1h").mean()
df_pt_1h = df_pt_1h.loc[start_date:end_date]


# --- KAN_U hourly data ---
kan_u = pd.read_csv("KAN_U_hour_new.csv", index_col=0, parse_dates=True)
kan_sel = kan_u.loc[start_date:end_date]


# ==================================================
# PLOT
# ==================================================

fig, ax1 = plt.subplots(figsize=(14, 6))

# Transparent background
fig.patch.set_alpha(0)
ax1.set_facecolor("none")

# --- Stage02 water depth (ax1) ---
ax1.plot(
    df_pt_1h.index,
    df_pt_1h[depth_col],
    linestyle="-",
    linewidth=1.5,
    color="white",
    label="Stage02 water depth"
)

# Secondary axis
ax2 = ax1.twinx()
ax2.set_facecolor("none")

# --- KAN_U air temperature (ax2) ---
ax2.plot(
    kan_sel.index,
    kan_sel["t_u"],
    linestyle="--",
    linewidth=1.5,
    color="white",
    label="KAN_U air temperature"
)

# ==================================================
# AXIS FORMATTING
# ==================================================

# Y labels
ax1.set_ylabel("Water depth (cm)", color="white")
ax2.set_ylabel("Air temperature (°C)", color="white")

# White ticks + spines
for ax in [ax1, ax2]:
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

# Grid
ax1.grid(True, color="white", alpha=0.2)

# ✅ X range
ax1.set_xlim(start_date, end_date)
ax1.set_ylim(130,190)

# ✅ DAILY ticks at midnight (FULL plot range)
xmin_plot, xmax_plot = ax1.get_xlim()
start = pd.to_datetime(mdates.num2date(xmin_plot)).normalize()
end   = pd.to_datetime(mdates.num2date(xmax_plot)).normalize()

daily_ticks = pd.date_range(start=start, end=end, freq="1D")

ax1.set_xticks(daily_ticks)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

plt.setp(ax1.get_xticklabels(), rotation=90, ha="center", color="white")

# ==================================================
# LEGENDS
# ==================================================

leg1 = ax1.legend(loc="upper left")
leg2 = ax2.legend(loc="upper right")

for leg in [leg1, leg2]:
    for text in leg.get_texts():
        text.set_color("white")
    leg.get_frame().set_facecolor("none")
    leg.get_frame().set_alpha(0.5)

# ==================================================
# SAVE
# ==================================================

plt.tight_layout()
plt.savefig("stage02_vs_temp.svg", transparent=True)
plt.show()

# %%
# ==================================================
# SCATTER PLOT: Water depth vs Air temperature
# ==================================================

fig, ax = plt.subplots(figsize=(8, 8))  # ✅ square figure

# Transparent background
fig.patch.set_alpha(0)
ax.set_facecolor("none")

# Align data on common timestamps
df_combined = pd.DataFrame({
    "depth": df_pt_1h[depth_col],
    "temp": kan_sel["t_u"]
}).dropna()

# --- Scatter (single style only) ---
ax.scatter(
    df_combined["temp"],
    df_combined["depth"],
    marker="o",
    facecolors="none",
    edgecolors="white",
    linewidths=0.8,
    label="Data"
)

## Linear reg
import numpy as np
from scipy.stats import pearsonr

# --- Linear regression ---
x = df_combined["temp"].values
y = df_combined["depth"].values

# fit line: y = m*x + b
m, b = np.polyfit(x, y, 1)

# correlation coefficient r and p-value
r, p = pearsonr(x, y)
p_txt = "p < 0.01" if p < 0.01 else f"p = {p:.3f}"

# line for plotting
x_line = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
y_line = m * x_line + b

ax.plot(
    x_line,
    y_line,
    color="white",
    linestyle="--",
    linewidth=1
)

# --- equation text ---
eq_text = f"y = {m:.2f}x + {b:.1f}\nr = {r:.2f}, {p_txt}, n = {len(x)}"

ax.text(
    0.05, 0.95,
    eq_text,
    transform=ax.transAxes,
    color="white",
    ha="left",
    va="top"
)

# ==================================================
# AXIS STYLE
# ==================================================

ax.set_xlabel("Air temperature (°C)", color="white")
ax.set_ylabel("Water depth (cm)", color="white")

# ✅ square plot area (important)
ax.set_box_aspect(1)

ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("white")

ax.grid(True, color="white", alpha=0.2)

ax.set_ylim(120, 200)
ax.set_xlim(-30, 10)

# ==================================================
# LEGEND
# ==================================================

#leg = ax.legend()

for text in leg.get_texts():
    text.set_color("white")

leg.get_frame().set_facecolor("none")
leg.get_frame().set_alpha(0.5)

# ==================================================
# SAVE
# ==================================================

plt.tight_layout()
plt.savefig("scatter_depth_vs_temp.svg", transparent=True)
plt.show()

# %%
# ==================================================
# USER SETTINGS
# ==================================================

break_date = pd.to_datetime("2025-08-06")

from scipy.stats import pearsonr

# colour map for the XS dates
date_cmap = "viridis"

# ==================================================
# SCATTER PLOT: Channel width vs Air temperature
# ==================================================

fig, ax = plt.subplots(figsize=(9, 8))

# Transparent background
fig.patch.set_alpha(0)
ax.set_facecolor("none")

# --------------------------------------------------
# Match XS measurements to nearest air temperature
# --------------------------------------------------

temp_at_xs = kan_u["t_u"].reindex(
    df_xs["datetime"],
    method="nearest"
)

df_combined = df_xs.copy()

df_combined["temp"] = temp_at_xs.to_numpy()

df_combined = df_combined.dropna()

print(df_combined.head())
print("Rows:", len(df_combined))

# --------------------------------------------------
# Split before / after break date
# --------------------------------------------------

df_before = df_combined[
    df_combined["datetime"] < break_date
].copy()

df_after = df_combined[
    df_combined["datetime"] >= break_date
].copy()


# --------------------------------------------------
# Scatter, coloured by date
# --------------------------------------------------

sc = ax.scatter(
    df_combined["temp"],
    df_combined["xs_length"],
    c=mdates.date2num(df_combined["datetime"]),
    cmap=date_cmap,
    marker="o",
    edgecolors="white",
    linewidths=0.5
)

# Colour bar ticks every 2 days, anchored at the break date
k = np.arange(
    (df_combined["datetime"].min() - break_date).days // 2 - 1,
    (df_combined["datetime"].max() - break_date).days // 2 + 2
)
cbar_ticks = mdates.date2num(break_date + pd.to_timedelta(2 * k, unit="D"))

# Colour bar with dates, break date marked
cbar = fig.colorbar(
    sc,
    ax=ax,
    ticks=cbar_ticks,
    format=mdates.DateFormatter("%Y-%m-%d"),
    fraction=0.046,
    pad=0.04
)
cbar.ax.axhline(
    mdates.date2num(break_date),
    color="white",
    linestyle="--",
    linewidth=1
)
cbar.set_label("Date", color="white")
cbar.ax.tick_params(colors="white")
cbar.outline.set_edgecolor("white")

# --------------------------------------------------
# Date labels
# --------------------------------------------------

for _, row in df_combined.iterrows():

    label = row["datetime"].strftime("%m-%d")

    ax.annotate(
        label,
        (row["temp"], row["xs_length"]),
        textcoords="offset points",
        xytext=(3, 3),
        fontsize=8,
        color="white"
    )

# --------------------------------------------------
# Regression helper
# --------------------------------------------------

def add_regression(df, color, label):

    if len(df) < 2:
        return np.nan, np.nan, np.nan, np.nan

    x = df["temp"].values
    y = df["xs_length"].values

    m, b = np.polyfit(x, y, 1)

    r, p = pearsonr(x, y)

    x_line = np.linspace(
        x.min(),
        x.max(),
        100
    )

    y_line = m * x_line + b

    ax.plot(
        x_line,
        y_line,
        linestyle="--",
        linewidth=1.5,
        color=color,
        label=label
    )

    return m, b, r, p

# --------------------------------------------------
# Regressions
# --------------------------------------------------

m1, b1, r1, p1 = add_regression(
    df_before,
    "cyan",
    f"Before {break_date.date()}"
)

m2, b2, r2, p2 = add_regression(
    df_after,
    "orange",
    f"After {break_date.date()}"
)

# --------------------------------------------------
# Regression statistics
# --------------------------------------------------

p1_txt = "p < 0.01" if p1 < 0.01 else f"p = {p1:.3f}"
p2_txt = "p < 0.01" if p2 < 0.01 else f"p = {p2:.3f}"

stats_text = (
    f"Before {break_date.date()}\n"
    f"y = {m1:.3f}x + {b1:.3f}\n"
    f"r = {r1:.2f}, {p1_txt}, n = {len(df_before)}\n\n"
    f"After {break_date.date()}\n"
    f"y = {m2:.3f}x + {b2:.3f}\n"
    f"r = {r2:.2f}, {p2_txt}, n = {len(df_after)}"
)

ax.text(
    0.05,
    0.95,
    stats_text,
    transform=ax.transAxes,
    color="white",
    ha="left",
    va="top",
    bbox=dict(
        facecolor="black",
        alpha=0.2,
        edgecolor="white"
    )
)

# --------------------------------------------------
# Axis styling
# --------------------------------------------------

ax.set_xlabel(
    "Air temperature (°C)",
    color="white"
)

ax.set_ylabel(
    "Channel width (m)",
    color="white"
)

ax.set_box_aspect(1)

ax.tick_params(colors="white")

for spine in ax.spines.values():
    spine.set_color("white")

ax.grid(
    True,
    color="white",
    alpha=0.2
)

# --------------------------------------------------
# Legend
# --------------------------------------------------

leg = ax.legend()

for text in leg.get_texts():
    text.set_color("white")

leg.get_frame().set_facecolor("none")
leg.get_frame().set_alpha(0.5)

# --------------------------------------------------
# Save
# --------------------------------------------------

plt.tight_layout()

# tight bbox: colour bar + square axes otherwise clip the y label
plt.savefig(
    "scatter_channel_width_vs_temp.svg",
    transparent=True,
    bbox_inches="tight"
)

plt.show()

# %% [markdown]
# # PDH (positive degree hours) metrics
#
# Same plots as with air temperature above, but with PDH from the hourly
# KAN_U data (hourly t_u clipped at 0, as in b02):
# - `PDH_24h`: PDH summed over the 24 h preceding each observation
# - `cum_PDH`: PDH accumulated since `pdh_ref_date`
# - `t_mean_24h`: mean air temperature over the same 24 h (reference)
#
# Note: `cum_PDH` only increases over the season, so its correlation with
# channel width largely reflects elapsed time. The p-values assume
# independent samples; daily XS and overlapping PDH windows are
# autocorrelated, so they are optimistic.

# %%
from river.config import bathy_dir, pts_dir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress

# ==================================================
# USER SETTINGS
# ==================================================

cam = "ilh-cam1-pt"

# old XS data for now; switch back to bathy_dir once the new XS data is done
xs_dir = bathy_dir# / "_old_data"

temp_dat = "KAN_U_hour_new.csv"

pdh_start = pd.to_datetime("2025-07-14 00:00")
pdh_end   = pd.to_datetime("2025-09-01 00:00")

break_date = pd.to_datetime("2025-08-06")

# colour map for the XS dates
date_cmap = "viridis"

# cum_PDH is zero at this date (start of melt season)
pdh_ref_date = pd.to_datetime("2025-05-01 00:00")

# window of PDH preceding each observation
pdh_window = "24h"

# gaps in hourly t_u of up to max_gap_h hours are linearly interpolated,
# longer gaps stay NaN (i.e. count as 0 PDH in cum_PDH)
max_gap_h = 3

pdh_col = f"PDH_{pdh_window}"
tmean_col = f"t_mean_{pdh_window}"

# ==================================================
# LOAD XS DATA
# ==================================================

cam_dir = xs_dir / cam
records = []

for f in sorted(cam_dir.glob(f"{cam}_xs_coord_*.csv")):

    try:
        timestamp = f.stem.replace(f"{cam}_xs_coord_", "")
        dt = pd.to_datetime(timestamp, format="%Y%m%d_%H%M%S")

        df_tmp = pd.read_csv(f, header=None)

        # distance between XS end points, works for relative (local)
        # and absolute (UTM) coordinates
        xs_length = float(np.hypot(
            df_tmp.iloc[1, 0] - df_tmp.iloc[0, 0],
            df_tmp.iloc[1, 1] - df_tmp.iloc[0, 1]
        ))

        records.append({
            "datetime": dt,
            "xs_length": xs_length
        })

    except Exception as e:
        print(f"Skipping {f.name}: {e}")

df_xs = (
    pd.DataFrame(records)
    .dropna()
    .sort_values("datetime")
    .reset_index(drop=True)
)

print("XS rows:", len(df_xs))

# ==================================================
# LOAD HOURLY TEMP DATA + PDH
# ==================================================

kan_u_h = pd.read_csv(temp_dat, index_col=0, parse_dates=True)

t_raw = kan_u_h["t_u"].asfreq("1h")

# only interpolate gaps of up to max_gap_h hours
gap_id = t_raw.notna().cumsum()
gap_len = t_raw.isna().groupby(gap_id).transform("sum")
t_u = (
    t_raw
    .interpolate(method="time", limit_area="inside")
    .where(t_raw.notna() | (gap_len <= max_gap_h))
)

print(
    f"t_u gaps filled: {int(t_raw.isna().sum() - t_u.isna().sum())}, "
    f"still missing: {int(t_u.isna().sum())}"
)

df_pdh = pd.DataFrame({"t_u": t_u})
df_pdh["PDH"] = df_pdh["t_u"].clip(lower=0)

# KAN_U timestamps label the start of the hourly mean, so the window
# [t - w, t) (closed="left") covers exactly the hours before t
df_pdh[pdh_col] = df_pdh["PDH"].rolling(pdh_window, closed="left").sum()
df_pdh[tmean_col] = df_pdh["t_u"].rolling(pdh_window, closed="left").mean()

# cumulative PDH since pdh_ref_date, also only counting hours before t
in_season = df_pdh.index >= pdh_ref_date
df_pdh["cum_PDH"] = (
    df_pdh["PDH"]
    .where(in_season, 0)
    .fillna(0)
    .cumsum()
    .shift(1, fill_value=0)
    .where(in_season)
)

# ==================================================
# MATCH XS MEASUREMENTS TO PDH
# ==================================================

pdh_at_xs = df_pdh.reindex(
    df_xs["datetime"],
    method="nearest",
    tolerance=pd.Timedelta("1h")
)

df_xs_pdh = pd.concat(
    [df_xs, pdh_at_xs[[pdh_col, tmean_col, "cum_PDH"]].reset_index(drop=True)],
    axis=1
).dropna()

# change in channel width since the previous day; NaN where the previous XS
# is not exactly one day earlier (gaps), so it matches the 24 h PDH window
df_xs_pdh["d_xs_length"] = df_xs_pdh["xs_length"].diff().where(
    df_xs_pdh["datetime"].diff() == pd.Timedelta("1D")
)

df_pdh_before = df_xs_pdh[df_xs_pdh["datetime"] < break_date].copy()
df_pdh_after = df_xs_pdh[df_xs_pdh["datetime"] >= break_date].copy()

print(df_xs_pdh)

# ==================================================
# REGRESSION HELPER
# ==================================================

def pdh_regress(x, y):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())

    if n < 3 or np.ptp(x[mask]) == 0:
        return {"n": n, "slope": np.nan, "intercept": np.nan,
                "r": np.nan, "r2": np.nan, "p": np.nan}

    res = linregress(x[mask], y[mask])

    return {"n": n, "slope": res.slope, "intercept": res.intercept,
            "r": res.rvalue, "r2": res.rvalue**2, "p": res.pvalue}

# ==================================================
# DATE COLOUR BAR HELPER
# ==================================================

def add_date_colorbar(fig, ax, sc, dates, **cbar_kw):

    # ticks every 2 days, anchored at the break date
    k = np.arange(
        (dates.min() - break_date).days // 2 - 1,
        (dates.max() - break_date).days // 2 + 2
    )

    cbar = fig.colorbar(
        sc,
        ax=ax,
        ticks=mdates.date2num(break_date + pd.to_timedelta(2 * k, unit="D")),
        format=mdates.DateFormatter("%Y-%m-%d"),
        **{"fraction": 0.046, "pad": 0.04, **cbar_kw}
    )

    # break date marked
    cbar.ax.axhline(
        mdates.date2num(break_date),
        color="white",
        linestyle="--",
        linewidth=1
    )

    cbar.set_label("Date", color="white")
    cbar.ax.tick_params(colors="white")
    cbar.outline.set_edgecolor("white")

    return cbar

# %%
# ==================================================
# PLOT: Channel width, PDH and cumulative PDH
# ==================================================

fig, ax1 = plt.subplots(figsize=(14, 6))

# Make background transparent
fig.patch.set_alpha(0)
ax1.set_facecolor("none")

ax1.plot(
    df_xs["datetime"],
    df_xs["xs_length"],
    linestyle="None",
    marker="o",
    markerfacecolor="white",
    markeredgecolor="white",
    label="Channel width"
)

# Axis styling (white)
ax1.set_xlabel("Date", color="white")
ax1.set_ylabel("Channel width (m)", color="white")

ax1.set_ylim(0, 25)
ax1.tick_params(colors="white")
for spine in ax1.spines.values():
    spine.set_color("white")

ax1.grid(True, color="white", alpha=0.2)

ax1.set_xlim(pdh_start, pdh_end)

# Daily ticks at 00:00 across FULL plot range
daily_ticks = pd.date_range(
    start=pdh_start.normalize(),
    end=pdh_end.normalize(),
    freq="1D"
)
ax1.set_xticks(daily_ticks)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.setp(ax1.get_xticklabels(), rotation=90, ha="center", color="white")

pdh_sel = df_pdh.loc[pdh_start:pdh_end]

# Secondary axis: PDH over preceding window (dashed)
ax2 = ax1.twinx()
ax2.set_facecolor("none")

ax2.plot(
    pdh_sel.index,
    pdh_sel[pdh_col],
    linestyle="--",
    linewidth=1,
    color="white",
    label=f"KAN_U PDH, preceding {pdh_window}"
)

ax2.set_ylim(bottom=0)
ax2.set_ylabel(f"PDH, preceding {pdh_window} (°C)", color="white")

# Third axis: cumulative PDH (dotted, outer right)
ax3 = ax1.twinx()
ax3.set_facecolor("none")
ax3.spines["right"].set_position(("axes", 1.08))

ax3.plot(
    pdh_sel.index,
    pdh_sel["cum_PDH"],
    linestyle=":",
    linewidth=1.5,
    color="white",
    label=f"KAN_U cumulative PDH since {pdh_ref_date.date()}"
)

ax3.set_ylim(bottom=0)
ax3.set_ylabel(
    f"Cumulative PDH since {pdh_ref_date.date()} (°C)",
    color="white"
)

for ax in [ax2, ax3]:
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

# One legend for all axes (white text), on top axis so lines don't cover it
handles, labels = [], []
for ax in [ax1, ax2, ax3]:
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l

leg = ax3.legend(handles, labels, loc="upper left")

for text in leg.get_texts():
    text.set_color("white")

leg.get_frame().set_facecolor("none")
leg.get_frame().set_alpha(0.5)

plt.tight_layout()

plt.savefig("channel-width_pdh_cum.svg", transparent=True)
plt.show()

# %%
# ==================================================
# PLOT: Channel width, PDH and cumulative PDH
# ==================================================

fig, ax1 = plt.subplots(figsize=(14, 6))

# Make background transparent
fig.patch.set_alpha(0)
ax1.set_facecolor("none")

ax1.plot(
    df_xs["datetime"],
    df_xs["xs_length"],
    linestyle="None",
    marker="o",
    markerfacecolor="white",
    markeredgecolor="white",
    label="Channel width"
)

# Axis styling (white)
ax1.set_xlabel("Date", color="white")
ax1.set_ylabel("Channel width (m)", color="white")

ax1.set_ylim(0, 25)
ax1.tick_params(colors="white")
for spine in ax1.spines.values():
    spine.set_color("white")

ax1.grid(True, color="white", alpha=0.2)

ax1.set_xlim(pdh_start, pdh_end)

# Daily ticks at 00:00 across FULL plot range
daily_ticks = pd.date_range(
    start=pdh_start.normalize(),
    end=pdh_end.normalize(),
    freq="1D"
)
ax1.set_xticks(daily_ticks)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.setp(ax1.get_xticklabels(), rotation=90, ha="center", color="white")

pdh_sel = df_pdh.loc[pdh_start:pdh_end]

# Secondary axis: PDH over preceding window (dashed)
ax2 = ax1.twinx()
ax2.set_facecolor("none")

ax2.plot(
    pdh_sel.index,
    pdh_sel[pdh_col],
    linestyle="--",
    linewidth=1,
    color="white",
    label=f"KAN_U PDH, preceding {pdh_window}"
)

ax2.set_ylim(bottom=0)
ax2.set_ylabel(f"PDH, preceding {pdh_window} (°C)", color="white")


ax2.tick_params(colors="white")
for spine in ax2.spines.values():
    spine.set_color("white")

# One legend for all axes (white text), on top axis so lines don't cover it
handles, labels = [], []
for ax in [ax1, ax2]:
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l

leg = ax2.legend(handles, labels, loc="upper left")

for text in leg.get_texts():
    text.set_color("white")

leg.get_frame().set_facecolor("none")
leg.get_frame().set_alpha(0.5)

plt.tight_layout()

plt.savefig("channel-width_pdh.svg", transparent=True)
plt.show()

# %%
# ==================================================
# SCATTER PLOT: Channel width vs PDH (preceding window)
# ==================================================

def scatter_width_pdh(
    x_col,
    xlabel,
    fname,
    stats_xy=(0.05, 0.95),
    stats_va="top",
    legend_loc="lower right",
    y_col="xs_length",
    ylabel="Channel width (m)"
):

    fig, ax = plt.subplots(figsize=(9, 8))

    # Transparent background
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    stats_lines = []

    # --------------------------------------------------
    # Scatter, coloured by date
    # --------------------------------------------------

    d = df_xs_pdh.dropna(subset=[x_col, y_col])

    sc = ax.scatter(
        d[x_col],
        d[y_col],
        c=mdates.date2num(d["datetime"]),
        cmap=date_cmap,
        marker="o",
        edgecolors="white",
        linewidths=0.5
    )

    add_date_colorbar(fig, ax, sc, d["datetime"])

    # --------------------------------------------------
    # Regression, before / after break date
    # --------------------------------------------------

    for df, color, label in [
        (df_pdh_before, "cyan", f"Before {break_date.date()}"),
        (df_pdh_after, "orange", f"After {break_date.date()}"),
    ]:

        df = df.dropna(subset=[x_col, y_col])

        s = pdh_regress(df[x_col], df[y_col])

        if np.isfinite(s["slope"]):

            x_line = np.linspace(df[x_col].min(), df[x_col].max(), 100)

            ax.plot(
                x_line,
                s["slope"] * x_line + s["intercept"],
                linestyle="--",
                linewidth=1.5,
                color=color,
                label=label
            )

        p_txt = "p < 0.01" if s["p"] < 0.01 else f"p = {s['p']:.3f}"

        stats_lines.append(
            f"{label}\n"
            f"y = {s['slope']:.4f}x + {s['intercept']:.3f}\n"
            f"r = {s['r']:.2f}, {p_txt}, n = {s['n']}"
        )

    # --------------------------------------------------
    # Date labels
    # --------------------------------------------------

    for _, row in d.iterrows():

        ax.annotate(
            row["datetime"].strftime("%m-%d"),
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=8,
            color="white"
        )

    # --------------------------------------------------
    # Regression statistics
    # --------------------------------------------------

    ax.text(
        *stats_xy,
        "\n\n".join(stats_lines),
        transform=ax.transAxes,
        color="white",
        ha="left",
        va=stats_va,
        bbox=dict(
            facecolor="black",
            alpha=0.2,
            edgecolor="white"
        )
    )

    # --------------------------------------------------
    # Axis styling
    # --------------------------------------------------

    ax.set_xlabel(xlabel, color="white")
    ax.set_ylabel(ylabel, color="white")

    ax.set_box_aspect(1)

    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    ax.grid(True, color="white", alpha=0.2)

    # --------------------------------------------------
    # Legend
    # --------------------------------------------------

    leg = ax.legend(loc=legend_loc)

    for text in leg.get_texts():
        text.set_color("white")

    leg.get_frame().set_facecolor("none")
    leg.get_frame().set_alpha(0.5)

    plt.tight_layout()
    # tight bbox: colour bar + square axes otherwise clip the y label
    plt.savefig(fname, transparent=True, bbox_inches="tight")
    plt.show()


scatter_width_pdh(
    pdh_col,
    f"PDH, preceding {pdh_window} (°C)",
    "scatter_channel_width_vs_pdh.svg"
)

# %%
# ==================================================
# SCATTER PLOT: Channel width vs cumulative PDH
# ==================================================

scatter_width_pdh(
    "cum_PDH",
    f"Cumulative PDH since {pdh_ref_date.date()} (°C)",
    "scatter_channel_width_vs_cum_pdh.svg",
    # width decreases with cum_PDH -> stats box and legend into the empty corners
    stats_xy=(0.05, 0.05),
    stats_va="bottom",
    legend_loc="upper right"
)

# %%
# ==================================================
# SCATTER PLOT: Change in channel width vs PDH (preceding window)
# ==================================================

scatter_width_pdh(
    pdh_col,
    f"PDH, preceding {pdh_window} (°C)",
    "scatter_d_channel_width_vs_pdh.svg",
    y_col="d_xs_length",
    ylabel="Change in channel width since previous day (m)",
    # points cluster around 0 on the left -> stats box and legend into the empty corners
    stats_xy=(0.05, 0.05),
    stats_va="bottom",
    legend_loc="upper left"
)

# %%
# ==================================================
# SCATTER PLOT: Change in channel width vs air temperature (preceding window)
# ==================================================

scatter_width_pdh(
    tmean_col,
    f"Mean air temperature, preceding {pdh_window} (°C)",
    "scatter_d_channel_width_vs_temp.svg",
    y_col="d_xs_length",
    ylabel="Change in channel width since previous day (m)",
    # points cluster around 0 on the left -> stats box and legend into the empty corners
    stats_xy=(0.05, 0.05),
    stats_va="bottom",
    legend_loc="upper left"
)

# %%
# ==================================================
# REGRESSION METRICS: Channel width vs PDH
# ==================================================
# Mean air temperature over the same window is included as reference,
# i.e. whether PDH explains channel width better than air temperature

pdh_predictors = [pdh_col, "cum_PDH", tmean_col]

pdh_subsets = {
    "all": df_xs_pdh,
    "before": df_pdh_before,
    "after": df_pdh_after,
}

df_pdh_stats = pd.DataFrame([
    {"x": x, "subset": name, **pdh_regress(df[x], df["xs_length"])}
    for x in pdh_predictors
    for name, df in pdh_subsets.items()
])

print(df_pdh_stats.round(4).to_string(index=False))

df_pdh_stats.to_csv("xs_pdh_regression.csv", index=False)

# %%
# ==================================================
# PDH WINDOW SENSITIVITY
# ==================================================
# Correlation of channel width with PDH summed over different preceding
# windows, i.e. over which time span the channel responds to melt forcing

pdh_windows_h = [6, 12, 24, 48, 72, 96, 120, 168]

masks = {
    "all": np.ones(len(df_xs_pdh), dtype=bool),
    "before": (df_xs_pdh["datetime"] < break_date).to_numpy(),
    "after": (df_xs_pdh["datetime"] >= break_date).to_numpy(),
}

rows = []

for w in pdh_windows_h:

    pdh_w = (
        df_pdh["PDH"]
        .rolling(f"{w}h", closed="left")
        .sum()
        .reindex(
            df_xs_pdh["datetime"],
            method="nearest",
            tolerance=pd.Timedelta("1h")
        )
        .to_numpy()
    )

    for name, mask in masks.items():
        rows.append({
            "window_h": w,
            "subset": name,
            **pdh_regress(pdh_w[mask], df_xs_pdh["xs_length"].to_numpy()[mask])
        })

df_pdh_win = pd.DataFrame(rows)

print(df_pdh_win.pivot(index="window_h", columns="subset", values="r").round(3))

df_pdh_win.to_csv("xs_pdh_window_sensitivity.csv", index=False)

# --------------------------------------------------
# Plot r vs window length
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

# Transparent background
fig.patch.set_alpha(0)
ax.set_facecolor("none")

for name, color, label in [
    ("all", "white", "All"),
    ("before", "cyan", f"Before {break_date.date()}"),
    ("after", "orange", f"After {break_date.date()}"),
]:

    d = df_pdh_win[df_pdh_win["subset"] == name]

    ax.plot(
        d["window_h"],
        d["r"],
        marker="o",
        linewidth=1,
        color=color,
        label=label
    )

ax.axhline(0, color="white", linewidth=0.8)

ax.set_xticks(pdh_windows_h)
ax.set_ylim(-1, 1)

ax.set_xlabel("PDH window preceding XS (h)", color="white")
ax.set_ylabel("r (channel width vs PDH)", color="white")

ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("white")

ax.grid(True, color="white", alpha=0.2)

leg = ax.legend(loc="lower right")

for text in leg.get_texts():
    text.set_color("white")

leg.get_frame().set_facecolor("none")
leg.get_frame().set_alpha(0.5)

plt.tight_layout()

plt.savefig("r_channel_width_vs_pdh_window.svg", transparent=True)
plt.show()

# %% [markdown]
# # Hysteresis and response time
#
# Three plots, one question each (daily width at 12:00 vs PDH, split at `break_date`):
#
# 1. **Is there a lag?** Trajectories of width vs PDH with arrows in time order.
#    Anticlockwise = width lags the forcing. With PDH that has memory (right panel)
#    the loop collapses when the memory matches the lag.
# 2. **How large is the lag?** r of width vs PDH with exponential memory τ
#    (hourly PDH before the image weighted by exp(-s/τ), expressed per day like
#    `PDH_24h`). The τ with the highest r is the response time; half of the
#    adjustment happens within τ·ln2.
# 3. **How sensitive is the width?** Width vs PDH with memory `hyst_tau`; the
#    slope per period is the sensitivity (m per °C of PDH).
#
# Width and PDH both decline through the season, so raw r and slopes include this
# seasonal trend (upper bound). The dashed curves / "trend removed" values remove a
# linear trend from width and PDH within each period (lower bound, as part of the
# real response declines with the season too). The truth lies in between.

# %%
# ==================================================
# USER SETTINGS
# ==================================================

# PDH memory (e-folding time, h) used in the trajectory and sensitivity plots
hyst_tau = 48

# e-folding times (h) tested in the memory scan
hyst_taus = [6, 12, 18, 24, 36, 48, 60, 72, 96, 120, 168, 240]

# ==================================================
# PDH WITH EXPONENTIAL MEMORY
# ==================================================
# weighted mean of hourly PDH over the hours before t (weights exp(-s/tau)),
# times 24 -> per day, comparable to PDH_24h; shift(1) keeps only hours before t

def pdh_ewm(tau_h):
    return (
        df_pdh["PDH"]
        .fillna(0)
        .ewm(alpha=1 - np.exp(-1 / tau_h), adjust=False)
        .mean()
        .shift(1)
        * 24
    )


ewm_col = f"PDH_ewm{hyst_tau}"

df_hyst = df_xs_pdh[["datetime", "xs_length", pdh_col]].copy().reset_index(drop=True)
df_hyst[ewm_col] = pdh_ewm(hyst_tau).reindex(df_hyst["datetime"]).to_numpy()
df_hyst["regime"] = np.where(df_hyst["datetime"] < break_date, "before", "after")

# days since first XS (for the seasonal trend)
df_hyst["t_day"] = (
    df_hyst["datetime"] - df_hyst["datetime"].iloc[0]
).dt.total_seconds() / 86400

regime_style = {
    "before": ("cyan", f"Before {break_date.date()}"),
    "after": ("orange", f"After {break_date.date()}"),
}

# ==================================================
# HELPERS
# ==================================================

def minmax(a):
    a = np.asarray(a, dtype=float)
    return (a - a.min()) / (a.max() - a.min())


def loop_area(x, y):
    # signed area of the (closed) path in min-max normalised space,
    # > 0 = anticlockwise (forcing to the right, width up) = width lags
    x, y = minmax(x), minmax(y)
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def detrend(a, t):
    # residuals of a linear fit against time (seasonal trend removed)
    a = np.asarray(a, dtype=float)
    return a - np.polyval(np.polyfit(t, a, 1), t)


print(df_hyst)

# %%
# ==================================================
# TRAJECTORY PLOTS: Channel width vs PDH (arrows in time order)
# ==================================================

fig, axes = plt.subplots(1, 2, figsize=(17, 8))

# Transparent background
fig.patch.set_alpha(0)

d_last = df_hyst[df_hyst["regime"] == "before"].iloc[-1]
d_first = df_hyst[df_hyst["regime"] == "after"].iloc[0]

for ax, x_col, xlabel in [
    (axes[0], pdh_col, f"PDH, preceding {pdh_window} (°C)"),
    (axes[1], ewm_col, f"PDH, exp. memory τ = {hyst_tau} h (°C)"),
]:

    ax.set_facecolor("none")

    area_lines = []

    # --------------------------------------------------
    # Arrows in time order, per regime
    # --------------------------------------------------

    for regime, (color, label) in regime_style.items():

        d = df_hyst[df_hyst["regime"] == regime]
        x = d[x_col].to_numpy()
        y = d["xs_length"].to_numpy()

        for i in range(len(d) - 1):
            ax.annotate(
                "",
                xy=(x[i + 1], y[i + 1]),
                xytext=(x[i], y[i]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    linewidth=1.2,
                    shrinkA=4,
                    shrinkB=4
                )
            )

        # legend entry
        ax.plot([], [], color=color, linewidth=1.2, label=label)

        area_lines.append(f"{label}: {loop_area(x, y):+.2f}")

    # gap between the regimes
    ax.annotate(
        "",
        xy=(d_first[x_col], d_first["xs_length"]),
        xytext=(d_last[x_col], d_last["xs_length"]),
        arrowprops=dict(
            arrowstyle="-|>",
            color="white",
            linewidth=1,
            linestyle="--",
            alpha=0.6,
            shrinkA=4,
            shrinkB=4
        )
    )
    ax.plot(
        [], [],
        color="white",
        linestyle="--",
        alpha=0.6,
        label=f"Gap {d_last['datetime']:%m-%d} → {d_first['datetime']:%m-%d} (no XS)"
    )

    # --------------------------------------------------
    # Points, coloured by date
    # --------------------------------------------------

    sc = ax.scatter(
        df_hyst[x_col],
        df_hyst["xs_length"],
        c=mdates.date2num(df_hyst["datetime"]),
        cmap=date_cmap,
        marker="o",
        edgecolors="white",
        linewidths=0.5,
        zorder=3
    )

    for _, row in df_hyst.iterrows():

        ax.annotate(
            row["datetime"].strftime("%m-%d"),
            (row[x_col], row["xs_length"]),
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=8,
            color="white"
        )

    # --------------------------------------------------
    # Loop area
    # --------------------------------------------------

    ax.text(
        0.97,
        0.05,
        "Signed loop area\n(> 0 anticlockwise = width lags)\n" + "\n".join(area_lines),
        transform=ax.transAxes,
        color="white",
        ha="right",
        va="bottom",
        bbox=dict(
            facecolor="black",
            alpha=0.2,
            edgecolor="white"
        )
    )

    # --------------------------------------------------
    # Axis styling
    # --------------------------------------------------

    ax.set_xlabel(xlabel, color="white")
    ax.set_ylabel("Channel width (m)", color="white")

    ax.set_box_aspect(1)

    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    ax.grid(True, color="white", alpha=0.2)

    leg = ax.legend(loc="upper left")

    for text in leg.get_texts():
        text.set_color("white")

    leg.get_frame().set_facecolor("none")
    leg.get_frame().set_alpha(0.5)

add_date_colorbar(fig, axes, sc, df_hyst["datetime"], shrink=0.8)

# no tight_layout here (not compatible with the shared colour bar); tight bbox trims the margins
plt.savefig("trajectory_channel_width_vs_pdh.svg", transparent=True, bbox_inches="tight")
plt.show()

# %%
# ==================================================
# MEMORY SCAN: r of channel width vs PDH with memory τ
# ==================================================

rows = []

for tau in hyst_taus:

    f_tau = pdh_ewm(tau).reindex(df_hyst["datetime"]).to_numpy()

    for regime in regime_style:

        m = (df_hyst["regime"] == regime).to_numpy()
        x = f_tau[m]
        y = df_hyst["xs_length"].to_numpy()[m]
        t = df_hyst["t_day"].to_numpy()[m]

        rows.append({
            "tau_h": tau,
            "regime": regime,
            "r": pdh_regress(x, y)["r"],
            # seasonal trend removed from width and PDH
            "r_detrended": pdh_regress(detrend(x, t), detrend(y, t))["r"],
        })

df_memory = pd.DataFrame(rows)

print(df_memory.pivot(index="tau_h", columns="regime", values=["r", "r_detrended"]).round(3))

df_memory.to_csv("xs_pdh_memory_scan.csv", index=False)

# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

# Transparent background
fig.patch.set_alpha(0)
ax.set_facecolor("none")

for regime, (color, label) in regime_style.items():

    d = df_memory[df_memory["regime"] == regime]

    best = d.loc[d["r"].idxmax()]
    ax.plot(
        d["tau_h"],
        d["r"],
        marker="o",
        linewidth=1.5,
        color=color,
        label=(
            f"{label}: max r = {best['r']:.2f} at τ = {best['tau_h']:.0f} h "
            f"(half response {best['tau_h'] * np.log(2):.0f} h)"
        )
    )

    best_d = d.loc[d["r_detrended"].idxmax()]
    ax.plot(
        d["tau_h"],
        d["r_detrended"],
        marker="o",
        markersize=4,
        linewidth=1,
        linestyle="--",
        color=color,
        label=f"{label}, trend removed: max r = {best_d['r_detrended']:.2f} at τ = {best_d['tau_h']:.0f} h"
    )

ax.axhline(0, color="white", linewidth=0.8)

ax.text(
    0.97,
    0.05,
    "solid: raw (includes seasonal trend, upper bound)\n"
    "dashed: seasonal trend removed (lower bound)",
    transform=ax.transAxes,
    color="white",
    ha="right",
    va="bottom",
    bbox=dict(
        facecolor="black",
        alpha=0.2,
        edgecolor="white"
    )
)

ax.set_xscale("log")
ax.set_xticks(hyst_taus)
ax.set_xticklabels([str(t) for t in hyst_taus])
ax.minorticks_off()
ax.set_ylim(-1, 1)

ax.set_xlabel("PDH memory, e-folding time τ (h)", color="white")
ax.set_ylabel("r (channel width vs PDH with memory τ)", color="white")

ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("white")

ax.grid(True, color="white", alpha=0.2)

leg = ax.legend(loc="lower left", fontsize=9)

for text in leg.get_texts():
    text.set_color("white")

leg.get_frame().set_facecolor("none")
leg.get_frame().set_alpha(0.5)

plt.tight_layout()
plt.savefig("memory_scan_channel_width_pdh.svg", transparent=True, bbox_inches="tight")
plt.show()

# %%
# ==================================================
# SCATTER PLOT: Channel width vs PDH with memory (sensitivity)
# ==================================================

fig, ax = plt.subplots(figsize=(9, 8))

# Transparent background
fig.patch.set_alpha(0)
ax.set_facecolor("none")

# --------------------------------------------------
# Scatter, coloured by date
# --------------------------------------------------

sc = ax.scatter(
    df_hyst[ewm_col],
    df_hyst["xs_length"],
    c=mdates.date2num(df_hyst["datetime"]),
    cmap=date_cmap,
    marker="o",
    edgecolors="white",
    linewidths=0.5
)

add_date_colorbar(fig, ax, sc, df_hyst["datetime"])

# --------------------------------------------------
# Regression per period + sensitivity
# --------------------------------------------------

stats_lines = []

for regime, (color, label) in regime_style.items():

    d = df_hyst[df_hyst["regime"] == regime]
    x = d[ewm_col].to_numpy()
    y = d["xs_length"].to_numpy()
    t = d["t_day"].to_numpy()

    s = pdh_regress(x, y)

    # slope with the seasonal trend held fixed (width ~ PDH + time)
    X = np.column_stack([np.ones_like(x), x, t])
    slope_trend = np.linalg.lstsq(X, y, rcond=None)[0][1]

    x_line = np.linspace(x.min(), x.max(), 100)

    ax.plot(
        x_line,
        s["slope"] * x_line + s["intercept"],
        linestyle="--",
        linewidth=1.5,
        color=color,
        label=label
    )

    p_txt = "p < 0.01" if s["p"] < 0.01 else f"p = {s['p']:.3f}"

    stats_lines.append(
        f"{label}: r = {s['r']:.2f}, {p_txt}, n = {s['n']}\n"
        f"sensitivity {s['slope']:.3f} m/°C ({100 * s['slope'] / y.mean():.1f} %/°C of mean width)\n"
        f"trend removed: {slope_trend:.3f} m/°C"
    )

# --------------------------------------------------
# Date labels
# --------------------------------------------------

for _, row in df_hyst.iterrows():

    ax.annotate(
        row["datetime"].strftime("%m-%d"),
        (row[ewm_col], row["xs_length"]),
        textcoords="offset points",
        xytext=(3, 3),
        fontsize=8,
        color="white"
    )

# --------------------------------------------------
# Regression statistics
# --------------------------------------------------

ax.text(
    0.97,
    0.05,
    "\n\n".join(stats_lines),
    transform=ax.transAxes,
    color="white",
    ha="right",
    va="bottom",
    fontsize=9,
    bbox=dict(
        facecolor="black",
        alpha=0.2,
        edgecolor="white"
    )
)

# --------------------------------------------------
# Axis styling
# --------------------------------------------------

ax.set_xlabel(f"PDH, exp. memory τ = {hyst_tau} h (°C)", color="white")
ax.set_ylabel("Channel width (m)", color="white")

ax.set_box_aspect(1)

ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("white")

ax.grid(True, color="white", alpha=0.2)

leg = ax.legend(loc="upper left")

for text in leg.get_texts():
    text.set_color("white")

leg.get_frame().set_facecolor("none")
leg.get_frame().set_alpha(0.5)

plt.tight_layout()
# tight bbox: colour bar + square axes otherwise clip the y label
plt.savefig("scatter_channel_width_vs_pdh_memory.svg", transparent=True, bbox_inches="tight")
plt.show()

# %%
# ==================================================
# BEST PDH MEMORY PER PERIOD (from the memory scan)
# ==================================================
# τ with the highest raw r for each period; before and after points then use
# their own memory instead of the common hyst_tau

best_tau = {
    regime: int(
        df_memory.loc[df_memory.loc[df_memory["regime"] == regime, "r"].idxmax(), "tau_h"]
    )
    for regime in regime_style
}

print("best τ per period (h):", best_tau)

df_best = df_hyst.copy()
df_best["PDH_best"] = np.nan

for regime, tau in best_tau.items():
    m = df_best["regime"] == regime
    df_best.loc[m, "PDH_best"] = pdh_ewm(tau).reindex(df_best.loc[m, "datetime"]).to_numpy()

best_xlabel = (
    f"PDH, best memory per period "
    f"(τ before = {best_tau['before']} h, τ after = {best_tau['after']} h) (°C)"
)

# ==================================================
# TRAJECTORY PLOT: Channel width vs PDH with best memory per period
# ==================================================

fig, ax = plt.subplots(figsize=(9, 8))

# Transparent background
fig.patch.set_alpha(0)
ax.set_facecolor("none")

area_lines = []

# --------------------------------------------------
# Arrows in time order, per regime
# --------------------------------------------------

for regime, (color, label) in regime_style.items():

    d = df_best[df_best["regime"] == regime]
    x = d["PDH_best"].to_numpy()
    y = d["xs_length"].to_numpy()

    for i in range(len(d) - 1):
        ax.annotate(
            "",
            xy=(x[i + 1], y[i + 1]),
            xytext=(x[i], y[i]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                linewidth=1.2,
                shrinkA=4,
                shrinkB=4
            )
        )

    # legend entry
    ax.plot([], [], color=color, linewidth=1.2, label=f"{label} (τ = {best_tau[regime]} h)")

    area_lines.append(f"{label}: {loop_area(x, y):+.2f}")

# gap between the regimes
b_last = df_best[df_best["regime"] == "before"].iloc[-1]
b_first = df_best[df_best["regime"] == "after"].iloc[0]

ax.annotate(
    "",
    xy=(b_first["PDH_best"], b_first["xs_length"]),
    xytext=(b_last["PDH_best"], b_last["xs_length"]),
    arrowprops=dict(
        arrowstyle="-|>",
        color="white",
        linewidth=1,
        linestyle="--",
        alpha=0.6,
        shrinkA=4,
        shrinkB=4
    )
)
ax.plot(
    [], [],
    color="white",
    linestyle="--",
    alpha=0.6,
    label=f"Gap {b_last['datetime']:%m-%d} → {b_first['datetime']:%m-%d} (no XS)"
)

# --------------------------------------------------
# Points, coloured by date
# --------------------------------------------------

sc = ax.scatter(
    df_best["PDH_best"],
    df_best["xs_length"],
    c=mdates.date2num(df_best["datetime"]),
    cmap=date_cmap,
    marker="o",
    edgecolors="white",
    linewidths=0.5,
    zorder=3
)

add_date_colorbar(fig, ax, sc, df_best["datetime"])

for _, row in df_best.iterrows():

    ax.annotate(
        row["datetime"].strftime("%m-%d"),
        (row["PDH_best"], row["xs_length"]),
        textcoords="offset points",
        xytext=(3, 3),
        fontsize=8,
        color="white"
    )

# --------------------------------------------------
# Loop area
# --------------------------------------------------

ax.text(
    0.97,
    0.05,
    "Signed loop area\n(> 0 anticlockwise = width lags)\n" + "\n".join(area_lines),
    transform=ax.transAxes,
    color="white",
    ha="right",
    va="bottom",
    bbox=dict(
        facecolor="black",
        alpha=0.2,
        edgecolor="white"
    )
)

# --------------------------------------------------
# Axis styling
# --------------------------------------------------

ax.set_xlabel(best_xlabel, color="white")
ax.set_ylabel("Channel width (m)", color="white")

ax.set_box_aspect(1)

ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("white")

ax.grid(True, color="white", alpha=0.2)

leg = ax.legend(loc="upper left")

for text in leg.get_texts():
    text.set_color("white")

leg.get_frame().set_facecolor("none")
leg.get_frame().set_alpha(0.5)

plt.tight_layout()
# tight bbox: colour bar + square axes otherwise clip the y label
plt.savefig("trajectory_channel_width_vs_pdh_best_tau.svg", transparent=True, bbox_inches="tight")
plt.show()

# %%
# ==================================================
# SCATTER PLOT: Channel width vs PDH with best memory per period (sensitivity)
# ==================================================

fig, ax = plt.subplots(figsize=(9, 8))

# Transparent background
fig.patch.set_alpha(0)
ax.set_facecolor("none")

# --------------------------------------------------
# Scatter, coloured by date
# --------------------------------------------------

sc = ax.scatter(
    df_best["PDH_best"],
    df_best["xs_length"],
    c=mdates.date2num(df_best["datetime"]),
    cmap=date_cmap,
    marker="o",
    edgecolors="white",
    linewidths=0.5
)

add_date_colorbar(fig, ax, sc, df_best["datetime"])

# --------------------------------------------------
# Regression per period + sensitivity
# --------------------------------------------------

stats_lines = []

for regime, (color, label) in regime_style.items():

    d = df_best[df_best["regime"] == regime]
    x = d["PDH_best"].to_numpy()
    y = d["xs_length"].to_numpy()
    t = d["t_day"].to_numpy()

    s = pdh_regress(x, y)

    # slope with the seasonal trend held fixed (width ~ PDH + time)
    X = np.column_stack([np.ones_like(x), x, t])
    slope_trend = np.linalg.lstsq(X, y, rcond=None)[0][1]

    x_line = np.linspace(x.min(), x.max(), 100)

    ax.plot(
        x_line,
        s["slope"] * x_line + s["intercept"],
        linestyle="--",
        linewidth=1.5,
        color=color,
        label=f"{label} (τ = {best_tau[regime]} h)"
    )

    p_txt = "p < 0.01" if s["p"] < 0.01 else f"p = {s['p']:.3f}"

    stats_lines.append(
        f"{label}: r = {s['r']:.2f}, {p_txt}, n = {s['n']}\n"
        f"sensitivity {s['slope']:.3f} m/°C ({100 * s['slope'] / y.mean():.1f} %/°C of mean width)\n"
        f"trend removed: {slope_trend:.3f} m/°C"
    )

# --------------------------------------------------
# Date labels
# --------------------------------------------------

for _, row in df_best.iterrows():

    ax.annotate(
        row["datetime"].strftime("%m-%d"),
        (row["PDH_best"], row["xs_length"]),
        textcoords="offset points",
        xytext=(3, 3),
        fontsize=8,
        color="white"
    )

# --------------------------------------------------
# Regression statistics
# --------------------------------------------------

ax.text(
    0.97,
    0.05,
    "\n\n".join(stats_lines),
    transform=ax.transAxes,
    color="white",
    ha="right",
    va="bottom",
    fontsize=9,
    bbox=dict(
        facecolor="black",
        alpha=0.2,
        edgecolor="white"
    )
)

# --------------------------------------------------
# Axis styling
# --------------------------------------------------

ax.set_xlabel(best_xlabel, color="white")
ax.set_ylabel("Channel width (m)", color="white")

ax.set_box_aspect(1)

ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("white")

ax.grid(True, color="white", alpha=0.2)

leg = ax.legend(loc="upper left")

for text in leg.get_texts():
    text.set_color("white")

leg.get_frame().set_facecolor("none")
leg.get_frame().set_alpha(0.5)

plt.tight_layout()
# tight bbox: colour bar + square axes otherwise clip the y label
plt.savefig("scatter_channel_width_vs_pdh_best_tau.svg", transparent=True, bbox_inches="tight")
plt.show()

# %%
# ==================================================
# COMBINED: hysteresis + linear regression
# left: PDH over preceding 24 h | right: PDH with memory τ (range)
# ==================================================

# ==================================================
# USER SETTINGS
# ==================================================

# PDH memory (h): lower, central, upper -> points at the central τ, bars and
# regression envelope over the range
tau_range = (36, 48, 60)

from scipy.stats import t as t_dist
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

tau_cols = [f"PDH_ewm{tau}" for tau in tau_range]

df_comb_h = df_hyst[["datetime", "xs_length", pdh_col, "regime", "t_day"]].copy()

for tau, col in zip(tau_range, tau_cols):
    df_comb_h[col] = pdh_ewm(tau).reindex(df_comb_h["datetime"]).to_numpy()

# ==================================================
# HELPERS
# ==================================================

def fit_stats(x, y, t):
    # raw regression + the same with the seasonal (linear time) trend removed
    s = pdh_regress(x, y)

    # slope with the trend held fixed (width ~ PDH + time)
    X = np.column_stack([np.ones_like(x), x, t])
    slope_trend = np.linalg.lstsq(X, y, rcond=None)[0][1]

    # partial r with the trend removed from both, p with n - 3 degrees of freedom
    r_trend = pdh_regress(detrend(x, t), detrend(y, t))["r"]
    dof = len(y) - 3
    p_trend = 2 * t_dist.sf(abs(r_trend) * np.sqrt(dof / (1 - r_trend**2)), dof)

    return {
        **s,
        "rel": 100 * s["slope"] / np.mean(y),
        "slope_trend": slope_trend,
        "r_trend": r_trend,
        "p_trend": p_trend,
        "area": loop_area(x, y),
    }


def p_range(ps):
    # central τ value with range over all τ
    if max(ps) < 0.01:
        return "p < 0.01"
    return f"p = {ps[1]:.3f} ({min(ps):.3f}–{max(ps):.3f})"


def val_range(vals, fmt):
    # central τ value with range over all τ
    return f"{fmt.format(vals[1])} ({fmt.format(min(vals))}–{fmt.format(max(vals))})"


def draw_arrows(ax, x, y, color):
    for i in range(len(x) - 1):
        ax.annotate(
            "",
            xy=(x[i + 1], y[i + 1]),
            xytext=(x[i], y[i]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                linewidth=1.2,
                shrinkA=4,
                shrinkB=4
            )
        )


# ==================================================
# PLOT
# ==================================================

fig, axes = plt.subplots(1, 2, figsize=(18, 10))

# Transparent background
fig.patch.set_alpha(0)

c_last = df_comb_h[df_comb_h["regime"] == "before"].iloc[-1]
c_first = df_comb_h[df_comb_h["regime"] == "after"].iloc[0]

for panel, ax in enumerate(axes):

    ax.set_facecolor("none")

    x_col = pdh_col if panel == 0 else tau_cols[1]

    handles, labels = [], []

    for regime, (color, label) in regime_style.items():

        d = df_comb_h[df_comb_h["regime"] == regime]
        y = d["xs_length"].to_numpy()
        t = d["t_day"].to_numpy()

        if panel == 0:

            # --------------------------------------------------
            # Left: PDH over preceding 24 h
            # --------------------------------------------------

            x = d[pdh_col].to_numpy()

            draw_arrows(ax, x, y, color)

            # straight line from the first to the last day of the period
            # (= closing line of the loop area)
            h_chord, = ax.plot(
                [x[0], x[-1]],
                [y[0], y[-1]],
                linestyle=":",
                linewidth=1.5,
                color=color
            )

            handles.append((Line2D([0], [0], color=color, linewidth=1.2), h_chord))
            labels.append(
                f"{label} (n = {len(y)})"
            )

        else:

            # --------------------------------------------------
            # Right: PDH with memory, points at central τ, range over τ
            # --------------------------------------------------

            X = d[tau_cols].to_numpy()
            x = X[:, 1]
            sts = [fit_stats(X[:, k], y, t) for k in range(len(tau_cols))]

            # PDH range over τ as horizontal bars
            ax.errorbar(
                x,
                y,
                xerr=[x - X.min(axis=1), X.max(axis=1) - x],
                fmt="none",
                ecolor=color,
                elinewidth=1,
                capsize=2,
                alpha=0.8,
                zorder=2
            )

            # regression envelope over τ + central line
            x_grid = np.linspace(X.min(), X.max(), 200)
            lines = np.array([s["slope"] * x_grid + s["intercept"] for s in sts])
            ax.fill_between(
                x_grid,
                lines.min(axis=0),
                lines.max(axis=0),
                color=color,
                alpha=0.15,
                linewidth=0
            )

            x_line = np.linspace(x.min(), x.max(), 100)
            h_line, = ax.plot(
                x_line,
                sts[1]["slope"] * x_line + sts[1]["intercept"],
                linestyle="--",
                linewidth=1.5,
                color=color
            )

            handles.append((Patch(color=color, alpha=0.15, linewidth=0), h_line))
            s48 = sts[1]  # fit at the middle τ of tau_range (48 h)
            p48 = "p < 0.01" if s48["p"] < 0.01 else f"p = {s48['p']:.2f}"
            p48_trend = "p < 0.01" if s48["p_trend"] < 0.01 else f"p = {s48['p_trend']:.2f}"

            labels.append(
                f"{label} (n = {s48['n']}):\n"
                f"Raw data:                         slope {s48['slope']:.2f} m/°C, r = {s48['r']:.2f}, {p48}\n"
                f"Seasonal trend removed: slope {s48['slope_trend']:.2f} m/°C, r = {s48['r_trend']:.2f}, {p48_trend}"
            )

    # --------------------------------------------------
    # Gap between the regimes (path panel only)
    # --------------------------------------------------

    if panel == 0:

        ax.annotate(
            "",
            xy=(c_first[x_col], c_first["xs_length"]),
            xytext=(c_last[x_col], c_last["xs_length"]),
            arrowprops=dict(
                arrowstyle="-|>",
                color="white",
                linewidth=1,
                linestyle="--",
                alpha=0.6,
                shrinkA=4,
                shrinkB=4
            )
        )
        handles.append(Line2D([0], [0], color="white", linestyle="--", alpha=0.6))
        labels.append(f"Inactive surface hydrology {c_last['datetime']:%Y-%m-%d} → {c_first['datetime']:%Y-%m-%d}")


    # --------------------------------------------------
    # Points, coloured by date
    # --------------------------------------------------

    sc = ax.scatter(
        df_comb_h[x_col],
        df_comb_h["xs_length"],
        c=mdates.date2num(df_comb_h["datetime"]),
        cmap=date_cmap,
        marker="o",
        edgecolors="white",
        linewidths=0.5,
        zorder=3
    )

    for _, row in df_comb_h.iterrows():

        ax.annotate(
            row["datetime"].strftime("%m-%d"),
            (row[x_col], row["xs_length"]),
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=8,
            color="white"
        )

    # --------------------------------------------------
    # Axis styling
    # --------------------------------------------------

    if panel == 0:
        ax.set_xlabel(f"PDH, preceding {pdh_window} (°C)", color="white")
    else:
        ax.set_xlabel(
            f"PDH, τ = {tau_range[1]}±{int(tau_range[1]-tau_range[0])}h (°C)",
            color="white"
        )
    ax.set_ylabel("Channel width (m)", color="white")

    ax.set_box_aspect(1)

    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    ax.grid(True, color="white", alpha=0.2)

    # --------------------------------------------------
    # Legend (below the panel, long labels)
    # --------------------------------------------------

    leg = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        fontsize=9,
        handler_map={tuple: HandlerTuple(ndivide=None)}
    )

    for text in leg.get_texts():
        text.set_color("white")

    leg.get_frame().set_facecolor("none")
    leg.get_frame().set_alpha(0.5)

add_date_colorbar(fig, axes, sc, df_comb_h["datetime"], shrink=0.6)

# no tight_layout here (not compatible with the shared colour bar); tight bbox trims the margins
plt.savefig("hysteresis_regression_channel_width_vs_pdh.svg", transparent=True, bbox_inches="tight")
plt.show()

# %%
# ==================================================
# USER SETTINGS
# ==================================================

pt_file = "Stage02-data-2026-04-24 10_40_48.csv"

pt_start = pd.to_datetime("2025-05-10 00:00")
pt_end   = pd.to_datetime("2025-07-17 00:00")

# ==================================================
# LOAD PT DATA
# ==================================================

df_pt = pd.read_csv(pts_dir / pt_file)

depth_col = [c for c in df_pt.columns if "Water_deep" in c][0]

df_pt["Time"] = pd.to_datetime(df_pt["Time"], errors="coerce")
df_pt[depth_col] = pd.to_numeric(df_pt[depth_col], errors="coerce")
df_pt = df_pt.dropna(subset=["Time"]).set_index("Time").sort_index()

# HOURLY RESAMPLING
df_pt_1h = df_pt[[depth_col]].resample("1h").mean().loc[pt_start:pt_end]

pdh_pt = df_pdh.loc[pt_start:pt_end]

# ==================================================
# PLOT: Stage02 water depth vs PDH
# ==================================================

fig, ax1 = plt.subplots(figsize=(14, 6))

# Transparent background
fig.patch.set_alpha(0)
ax1.set_facecolor("none")

# --- Stage02 water depth (ax1) ---
ax1.plot(
    df_pt_1h.index,
    df_pt_1h[depth_col],
    linestyle="-",
    linewidth=1.5,
    color="white",
    label="Stage02 water depth"
)

# Secondary axis
ax2 = ax1.twinx()
ax2.set_facecolor("none")

# --- KAN_U PDH over preceding window (ax2) ---
ax2.plot(
    pdh_pt.index,
    pdh_pt[pdh_col],
    linestyle="--",
    linewidth=1.5,
    color="white",
    label=f"KAN_U PDH, preceding {pdh_window}"
)

# Y labels
ax1.set_ylabel("Water depth (cm)", color="white")
ax2.set_ylabel(f"PDH, preceding {pdh_window} (°C)", color="white")

# White ticks + spines
for ax in [ax1, ax2]:
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

# Grid
ax1.grid(True, color="white", alpha=0.2)

ax1.set_xlim(pt_start, pt_end)
ax1.set_ylim(130, 190)
# headroom above the PDH peak for the legend
ax2.set_ylim(0, pdh_pt[pdh_col].max() * 1.2)

# DAILY ticks at midnight (FULL plot range)
daily_ticks = pd.date_range(
    start=pt_start.normalize(),
    end=pt_end.normalize(),
    freq="1D"
)
ax1.set_xticks(daily_ticks)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.setp(ax1.get_xticklabels(), rotation=90, ha="center", color="white")

# Legends
leg1 = ax1.legend(loc="upper left")
leg2 = ax2.legend(loc="upper right")

for leg in [leg1, leg2]:
    for text in leg.get_texts():
        text.set_color("white")
    leg.get_frame().set_facecolor("none")
    leg.get_frame().set_alpha(0.5)

plt.tight_layout()

plt.savefig("stage02_vs_pdh.svg", transparent=True)
plt.show()

# %%
# ==================================================
# SCATTER PLOT: Water depth vs PDH
# ==================================================

fig, ax = plt.subplots(figsize=(8, 8))

# Transparent background
fig.patch.set_alpha(0)
ax.set_facecolor("none")

# Align data on common timestamps
df_pt_pdh = pd.DataFrame({
    "depth": df_pt_1h[depth_col],
    "pdh": pdh_pt[pdh_col]
}).dropna()

ax.scatter(
    df_pt_pdh["pdh"],
    df_pt_pdh["depth"],
    marker="o",
    facecolors="none",
    edgecolors="white",
    linewidths=0.8
)

# --- Linear regression ---
s = pdh_regress(df_pt_pdh["pdh"], df_pt_pdh["depth"])

x_line = np.linspace(df_pt_pdh["pdh"].min(), df_pt_pdh["pdh"].max(), 100)

ax.plot(
    x_line,
    s["slope"] * x_line + s["intercept"],
    color="white",
    linestyle="--",
    linewidth=1
)

p_txt = "p < 0.01" if s["p"] < 0.01 else f"p = {s['p']:.3f}"

ax.text(
    0.05, 0.95,
    f"y = {s['slope']:.3f}x + {s['intercept']:.1f}\n"
    f"r = {s['r']:.2f}, {p_txt}, n = {s['n']}",
    transform=ax.transAxes,
    color="white",
    ha="left",
    va="top"
)

# Axis styling
ax.set_xlabel(f"PDH, preceding {pdh_window} (°C)", color="white")
ax.set_ylabel("Water depth (cm)", color="white")

ax.set_box_aspect(1)

ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("white")

ax.grid(True, color="white", alpha=0.2)

ax.set_ylim(120, 200)

plt.tight_layout()

plt.savefig("scatter_depth_vs_pdh.svg", transparent=True)
plt.show()

# %%
