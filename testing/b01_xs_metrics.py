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

# ==================================================
# USER SETTINGS
# ==================================================

cam = "ilh-cam1-pt"

pt_files = [
    "Stage01-data-2026-04-24 10_36_57.csv",
    "Stage02-data-2026-04-24 10_40_48.csv",
]

# --------------------------------------------------
# choose reference for time window:
# "xs" = align to XS coverage
# "pt" = align to PT coverage
# --------------------------------------------------
time_reference = "pt"   # or "pt"

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

        xs_length = abs(float(df_tmp.iloc[0, 0])) + abs(float(df_tmp.iloc[1, 0]))

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

ax1.plot(
    df_xs["datetime"],
    df_xs["xs_length"],
    "-o",
    color="black",
    linewidth=2,
    label="XS length"
)

ax1.set_xlabel("Date")
ax1.set_ylabel("XS Length")
ax1.grid(True)

ax2 = ax1.twinx()

colors = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange"]

# ==================================================
# PLOT PT DATA (2H RESAMPLED, ALIGNED TO MIDNIGHT)
# ==================================================

for i, (pt_file, df_pt, depth_col) in enumerate(pt_data):

    print("\nPT FILE:", pt_file)

    df_pt = df_pt.set_index("Time").sort_index()

    # 2-hour resampling anchored at start of day (00:00)
    df_pt_2h = (
        df_pt
        .resample("2h", origin="start_day", label="left", closed="left")
        .mean()
        .reset_index()
    )

    # restrict to chosen window
    df_pt_2h = df_pt_2h[
        (df_pt_2h["Time"] >= xmin) &
        (df_pt_2h["Time"] <= xmax)
    ]

    if df_pt_2h.empty:
        print("No data in window:", pt_file)
        continue

    ax2.plot(
        df_pt_2h["Time"],
        df_pt_2h[depth_col],
        "-o",
        linewidth=1.5,
        markersize=4,
        color=colors[i % len(colors)],
        label=pt_file
    )

# ==================================================
# FINALIZE
# ==================================================

ax2.set_ylabel("Water Depth (cm)")

ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.title(f"{cam}: XS Length vs Water Depth (2H aligned)")
plt.tight_layout()
plt.show()

# %%
df_pt

# %%
