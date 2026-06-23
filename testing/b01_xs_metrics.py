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

# ==================================================
# USER SETTINGS
# ==================================================

cam = "ilh-cam1-pt"

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

cam_dir = bathy_dir / cam
records = []

for f in sorted(cam_dir.glob(f"{cam}_xs_coord_*.csv")):

    try:
        timestamp = f.stem.replace(f"{cam}_xs_coord_", "")
        dt = pd.to_datetime(timestamp, format="%Y%m%d_%H%M%S")

        df_tmp = pd.read_csv(f, header=None)
        print(df_tmp)

        xs_length = abs(float(df_tmp.iloc[1, 0]-df_tmp.iloc[0, 0]))

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

plt.savefig("channel-width_pt_1.svg")
plt.show()

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

cam_dir = bathy_dir / cam
records = []

for f in sorted(cam_dir.glob(f"{cam}_xs_coord_*.csv")):

    try:
        timestamp = f.stem.replace(f"{cam}_xs_coord_", "")
        dt = pd.to_datetime(timestamp, format="%Y%m%d_%H%M%S")

        df_tmp = pd.read_csv(f, header=None)
        print(df_tmp)

        xs_length = abs(float(df_tmp.iloc[1, 0]-df_tmp.iloc[0, 0]))

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

# --- Linear regression ---
x = df_combined["temp"].values
y = df_combined["depth"].values

# fit line: y = m*x + b
m, b = np.polyfit(x, y, 1)

# correlation coefficient r
r = np.corrcoef(x, y)[0, 1]

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
eq_text = f"y = {m:.2f}x + {b:.1f}\nr = {r:.2f}"

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
