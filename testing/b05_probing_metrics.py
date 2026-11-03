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

# %% [markdown]
# # Probing profile 2026-04-23 on the left stake line (GCP4 → GCP3)
#
# Snow probing on 2026-04-23 (`data/field/ilhh/probing_ilhh.csv`: `distance_m`, `depth_cm`, `note`; every 0.5 m
# from 0 to 64 m) along the left stake line of the 2025-04-26 installation. The tape starts on the camera line, 10 m
# before pt1 (= GCP4); the sheet notes pt03 at 24 m and pt07 at 52 m. Where the probe did not get through
# ("impermeabl") there is no depth.
#
# Field sketch → our GCP numbering: sketch gcp1 = GCP4 (pt1), gcp2 = GCP1 (pt2), gcp3 = GCP3 (pt5),
# gcp4 = GCP2 (pt6). Stakes every 7 m from 10 m in front of the camera, alternating between the left (odd) and the
# right (even) line, 10 m apart: GCP4, pt3, GCP3 and pt7 are 14 m apart on the left line.
#
# 1. **Horizontal placement**: tape distance d → GCP4 + (d − 10) · unit(GCP4 → GCP3), in UTM and in the local GCP
#    system (as a07: GCP1 origin, s along GCP1 → GCP2)
# 2. **Surfaces** along the line and at the GCP4 pole (m below the pole top): 2025-04-26 installation (field
#    emergence), 2025-07-01 (emergence calibration), end of the 2025 melt (b02 central GCP model), 2026-04-23 (winter
#    surface rise at KAN_U / KAN_M) and the probed ice (2026-04-23 surface − probed depth)

# %%
from river.config import gcps_dir, frames_dir, aws_dir, data_dir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# %% [markdown]
# ## Load data

# %%
# ==================================================
# USER SETTINGS
# ==================================================
cam = "ilh-cam1-pt"
cam_dir = gcps_dir / cam
mod_dir = cam_dir / "_modelling"
probe_file = data_dir / "field" / "ilhh" / "probing_ilhh.csv"

D_GCP4 = 10.0                                       # tape position of GCP4 (pt1); the tape starts 10 m before it
MARKS = [("start", 0), ("GCP4 / pt1", 10), ("pt3", 24), ("GCP3 / pt5", 38), ("pt7", 52), ("end", 64)]
SKETCH = {1: "pt2", 2: "pt6", 3: "pt5", 4: "pt1"}   # our GCP -> stake of the field sketch
TAPE_FRAMES = ["20250701_120000", "20250820_160000"]    # frames for the tape overlay
SURF_FRAME = "20250701_120000"                      # emergence calibration image

EMERG_0426 = 0.72          # GCP4 emergence measured at the installation, 2025-04-26 (m)
SPRING_AWS = 0.081         # lowering 04-26 -> 07-01 estimated from KAN_U / KAN_M (PDH position, melt − accumulation), m
L_TEMP = 0.545             # air-temperature estimate of the lowering 07-01 -> 09-20 (b02 section 5), m
DDF = 0.0123               # m per °C·d, camera-site snow factor (07-01 -> 07-17), for the melt after the image record
T0 = pd.Timestamp("2025-07-01 12:00")               # first frame / calibration image
T_END = pd.Timestamp("2025-09-20 12:00")            # last frame of the GCP model
T_PROBE = pd.Timestamp("2026-04-23 10:00")          # probing day (camera logger ends 11:30)
ELEV = {"KAN_M": 1265, "site": 1590, "KAN_U": 1848}     # m a.s.l. (site estimated)
AWS_SENSORS = {"KAN_U": ("KAN_U_hour_new.csv", ["z_surf_combined", "z_boom_cor_u", "z_boom_cor_l"]),
               "KAN_M": ("KAN_M_hour_new.csv", ["z_surf_combined", "z_boom_cor_u", "z_stake_cor"])}   # first = central

# probing sheet (one header line above the column names)
print(probe_file.read_text().splitlines()[0])
pr = pd.read_csv(probe_file, skiprows=1)
ok = pr.depth_cm.notna()

real = pd.read_csv(cam_dir / f"{cam}_gcps_real.csv", header=None).values[:4, :2].astype(float)
G = {i + 1: real[i] for i in range(4)}
cal = pd.read_csv(cam_dir / f"{cam}_emergence_calibration.csv").iloc[0]
K_CAL = float(cal.px_per_m)
frames = pd.read_parquet(frames_dir / "_frame_paths.parquet")
env = pd.read_csv(mod_dir / "gcp_model_envelope.csv", parse_dates=["timestamp"])


def read_gcps(path):
    return pd.read_csv(path, header=None).values[:4, :2].astype(float)


def model_gcps(stamp, gcp_set="central"):
    # modelled GCP image coordinates: same timestamp, else the first file of that date (as in a05 / a07)
    folder = mod_dir / f"output_{gcp_set}"
    f = folder / f"{cam}_gcps_img_{stamp}.csv"
    if not f.exists():
        f = sorted(folder.glob(f"{cam}_gcps_img_{stamp[:8]}_*.csv"))[0]
    return read_gcps(f)


def frame_image(stamp):
    date, time_ = stamp.split("_")
    row = frames[(frames.camera == cam) & (frames.date_yyyymmdd == date) & (frames.time_hhmmss == time_)]
    return mpimg.imread(row.frame_path.iloc[0])


def homography(gcps_img):
    # ground -> image homography of the GCPs (DLT as in a05) and the a05 image scale (px per m across the view)
    wm, im = real.mean(0), gcps_img.mean(0)
    a = []
    for (X, Y), (u, v) in zip(real - wm, gcps_img - im):
        a += [[X, Y, 1, 0, 0, 0, -u * X, -u * Y, -u], [0, 0, 0, X, Y, 1, -v * X, -v * Y, -v]]
    h = np.linalg.svd(np.array(a))[2][-1].reshape(3, 3)
    g = np.linalg.inv(h)

    def to_img(p):
        p = np.atleast_2d(p)
        q = (h @ np.c_[p - wm, np.ones(len(p))].T).T
        return im + q[:, :2] / q[:, 2:]

    def scale(x, y):
        q = g @ np.array([x - im[0], y - im[1], 1.0])
        d = g[:, 0]
        return 1 / np.hypot((d[0] * q[2] - q[0] * d[2]) / q[2] ** 2, (d[1] * q[2] - q[1] * d[2]) / q[2] ** 2)

    return to_img, scale


def surface_line(gcps_img, x):
    # image y of the line through GCP3 and GCP4 (as in a05)
    (xa, ya), (xb, yb) = gcps_img[2], gcps_img[3]
    return ya + (yb - ya) / (xb - xa) * (np.asarray(x, dtype=float) - xa)


# %%
# figures: saved to FIG_ROOT/<theme>/<name>.png
from river.config import fig_dir

FIG_ROOT = fig_dir                                  # results/figures (FIG_DIR in setup.sh)
FIG_DPI = 200


def save_fig(fig, theme, name):
    out = FIG_ROOT / theme / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")

# %% [markdown]
# ## 1. Horizontal placement on the GCP4 → GCP3 line
#
# Surveyed GCP distances (`gcps_real`) against the nominal sketch layout, then the probing tape on the line through
# GCP4 and GCP3 (GCP4 at 10 m, 14 m steps), in UTM and in the local GCP system.

# %%
# nominal sketch positions: (along from the camera line, across; + = left line)
nominal = {f"pt{k}": (10 + 7 * (k - 1), 5.0 if k % 2 else -5.0) for k in range(1, 9)}
rows = []
for a, b in [(1, 2), (2, 3), (3, 4), (4, 1), (1, 3), (2, 4)]:
    pa, pb = nominal[SKETCH[a]], nominal[SKETCH[b]]
    rows.append({"pair": f"GCP{a}-GCP{b}", "sketch": f"{SKETCH[a]}-{SKETCH[b]}",
                 "surveyed_m": np.linalg.norm(G[a] - G[b]), "nominal_m": np.hypot(pa[0] - pb[0], pa[1] - pb[1])})
check = pd.DataFrame(rows)
check["diff_m"] = check.surveyed_m - check.nominal_m
print(check.round(2).to_string(index=False))

# local system (as a07): GCP1 origin, s along GCP1 -> GCP2, t across (+ = left)
u12 = (G[2] - G[1]) / np.linalg.norm(G[2] - G[1])
n12 = np.array([-u12[1], u12[0]])
L43 = np.linalg.norm(G[3] - G[4])
u43 = (G[3] - G[4]) / L43
n43 = np.array([-u43[1], u43[0]])


def to_st(p):
    d = np.atleast_2d(p) - G[1]
    return d @ u12, d @ n12


def on_line(d, offset=0.0):
    # ground point at tape distance d on the GCP4 -> GCP3 line (offset: across, + = left)
    d = np.atleast_1d(d)
    return G[4] + np.outer(d - D_GCP4, u43) + np.outer(np.broadcast_to(offset, d.shape), n43)


xy = on_line(pr.distance_m.values)
pr["E"], pr["N"] = xy[:, 0], xy[:, 1]
pr["s_local"], pr["t_local"] = to_st(xy)


def bearing(v):
    return np.degrees(np.arctan2(v[0], v[1])) % 360


print(f"\nGCP4 -> GCP3 {L43:.2f} m (nominal 28): GCP3 at tape {D_GCP4 + L43:.2f} m; "
      f"bearing {bearing(u43):.1f}° vs GCP1 -> GCP2 {bearing(u12):.1f}°")
for k in (4, 3):
    s, t = to_st(G[k])
    print(f"GCP{k}: s {s[0]:.2f} m, t {t[0]:.2f} m (distance to the right line, nominal 10 m)")
for name, d_note, n in [("pt03", 24, 1), ("pt07", 52, 3)]:
    print(f"{name} noted at {d_note} m -> GCP4 at {d_note - 14 * n:.2f} m (14 m steps) "
          f"or {d_note - L43 / 2 * n:.2f} m (surveyed steps {L43 / 2:.2f} m)")
pr.head()

# %%
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.plot(pr.s_local[ok], pr.t_local[ok], ".", c="steelblue", ms=4, label="probing (every 0.5 m)")
for lab, d in MARKS:
    s, t = to_st(on_line(d))
    ax.plot(s, t, "|", c="k", ms=14, mew=1.5)
    ax.annotate(f"{lab}\n{d} m", (s[0], t[0]), (0, 12), textcoords="offset points", ha="center", fontsize=8)
for k, p in G.items():
    s, t = to_st(p)
    ax.plot(s, t, "o", mfc="none", mec="magenta", ms=10, mew=1.5)
    ax.annotate(f"GCP{k}", (s[0], t[0]), (8, -14), textcoords="offset points", color="magenta", fontweight="bold")
for k in (2, 4, 6, 8):                               # right line: 14 m steps from GCP1 (pt2)
    s, t = to_st(G[1] + 7 * (k - 2) * u12)
    ax.plot(s, t, "s", mfc="none", mec="0.5", ms=6)
    ax.annotate(f"pt{k}", (s[0], t[0]), (0, -14), textcoords="offset points", ha="center", fontsize=8, color="0.5")
s3, t3 = to_st(G[3])
ax.annotate(f"surveyed GCP3 {L43 - 28:.2f} m beyond the 38 m mark", (s3[0], t3[0]), (25, 25), textcoords="offset points",
            fontsize=8, color="magenta", arrowprops=dict(arrowstyle="->", color="magenta", lw=0.8))
ax.set_aspect("equal")
ax.set_ylim(-3, 14)
ax.set_xlabel("s along GCP1 → GCP2 (m)")
ax.set_ylabel("t across, + = left (m)")
ax.set_title("Probing on the GCP4 → GCP3 (left stake) line, local GCP system")
ax.grid(alpha=0.3)
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
save_fig(fig, "probing_alignment", "probing_plan_view_local")
plt.show()

# %%
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(pr.distance_m, pr.depth_cm, "-", c="steelblue", lw=1.2)          # gaps where no depth
y_top, y_bottom = pr.depth_cm.min() - 5, pr.depth_cm.max() + 7
for lab, d in MARKS[1:-1]:
    ax.axvline(d, c="k", lw=1, alpha=0.6)
    ax.text(d, y_bottom, lab, rotation=90, va="bottom", ha="right", fontsize=8)
ax.axvline(D_GCP4 + L43, c="magenta", lw=1, ls=":")
ax.text(D_GCP4 + L43, y_top, " surveyed GCP3", color="magenta", fontsize=8, va="top")
ax.set_ylim(y_bottom, y_top)
ax.set_xlim(-1, 65)
ax.set_xlabel("probing distance from the camera line (m)")
ax.set_ylabel("depth below the snow surface (cm)")
ax.set_title("Probed depth 2026-04-23")
ax.grid(alpha=0.3)
fig.tight_layout()
save_fig(fig, "probing_alignment", "probing_depth_profile")
plt.show()

# %% [markdown]
# The probing tape on the camera frames (central GCP model of each frame). Below 10 m and beyond GCP3 the positions
# are extrapolated outside the GCPs; beyond GCP3 the line is squeezed into few pixels.

# %%
for stamp in TAPE_FRAMES:
    img = frame_image(stamp)
    gi = model_gcps(stamp)
    to_img, _ = homography(gi)
    uv = to_img(xy)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.plot(uv[ok, 0], uv[ok, 1], ".", c="yellow", ms=4)
    for lab, d in MARKS:
        p = to_img(on_line(d))[0]
        ax.plot(*p, "|", c="w", ms=16, mew=2)
        ax.annotate(f"{lab} ({d})", p, (-6, -14), textcoords="offset points", ha="right", fontsize=8, color="w")
    for k in range(4):
        ax.plot(*gi[k], "o", mfc="none", mec="magenta", ms=11, mew=1.5)
        ax.annotate(f"GCP{k + 1}", gi[k], (9, 10), textcoords="offset points", color="magenta", fontsize=9)
    zz = to_img(on_line(np.array([6.0, 56.0])))
    ax.set_xlim(zz[:, 0].min() - 150, zz[:, 0].max() + 250)
    ax.set_ylim(zz[:, 1].max() + 150, zz[:, 1].min() - 120)
    ax.set_title(f"{cam} {stamp}: probing tape on the GCP4–GCP3 line")
    ax.axis("off")
    fig.tight_layout()
    save_fig(fig, "probing_alignment", f"probing_tape_{stamp}")
    plt.show()

# %% [markdown]
# Stakes on the calibration image: a ground grid around the 24 m and 52 m marks (cyan every 0.5 m along the tape,
# numbers every 1 m; orange every 0.5 m across, red = the line). 1 m along the line is about 11 px at 24 m and
# 3 px at 52 m.

# %%
img = frame_image(SURF_FRAME)
gi07 = model_gcps(SURF_FRAME)
to_img07, scale07 = homography(gi07)
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
for ax, (stake, dc) in zip(axs, [("pt3", 24), ("pt7", 52)]):
    D = np.arange(dc - 4, dc + 4.01, 0.5)
    O = np.arange(-2, 2.01, 0.5)
    ax.imshow(img, interpolation="nearest")
    for o in O:
        p = to_img07(on_line(D, o))
        ax.plot(p[:, 0], p[:, 1], "-", c="red" if o == 0 else "orange", lw=1.2 if o == 0 else 0.5)
    for dd in D:
        p = to_img07(on_line(np.full(len(O), dd), O))
        major = dd == round(dd)
        ax.plot(p[:, 0], p[:, 1], "-", c="cyan", lw=1.0 if major else 0.4)
        if major:
            ax.annotate(f"{dd:.0f}", p[0], (0, -10), textcoords="offset points", fontsize=7, color="cyan", ha="center")
    c = to_img07(on_line(D[[0, -1, 0, -1]], O[[0, 0, -1, -1]]))
    ax.set_xlim(c[:, 0].min() - 25, c[:, 0].max() + 25)
    ax.set_ylim(c[:, 1].max() + 20, c[:, 1].min() - 60)
    ax.set_title(f"{SURF_FRAME}: {stake} at {dc} m")
    ax.axis("off")
fig.tight_layout()
save_fig(fig, "probing_alignment", "probing_stakes_pt3_pt7")
plt.show()

# %% [markdown]
# ## 2. Surfaces along the line and at the GCP4 pole
#
# All surfaces hang from the 2025-07-01 surface of the manual picks (GCP4 base = calibration base, 0.93 m below the
# pole top) and are shifted by the same metres along the whole line; metres → px with the a05 vertical scale
# (calibration px/m at GCP4 × image-scale ratio of the homography):
# - **2025-04-26 installation**: field emergence 0.72 m; dashed: AWS estimate of the spring lowering
# - **end of the 2025 melt**: GCP4 lowering 07-01 → 09-20 of the central GCP model (b02) + melt after 09-20
#   (camera-logger PDH × snow factor)
# - **2026-04-23**: end of melt − winter surface rise 09-20 → 2026-04-23 (KAN_U / KAN_M, weighted by elevation)
# - **probed ice**: 2026-04-23 surface − probed depth (no line where no depth)

# %%
# melt after the image record (camera logger: 10-min values labelled at the interval end -> hourly means)
lg = pd.read_csv(aws_dir / "hydro_ilh_Table1.dat", skiprows=[0, 2, 3], na_values=["NAN"])
lg["time"] = pd.to_datetime(lg.TIMESTAMP) - pd.Timedelta("10min")
t_cam = lg.set_index("time").AirTemp_C.resample("1h").mean()
pdh_post = t_cam[T_END:T_PROBE].clip(lower=0).sum() / 24
melt_post = DDF * pdh_post
print(f"camera logger after {T_END:%Y-%m-%d}: {pdh_post:.2f} °C·d above 0 -> melt {melt_post * 100:.1f} cm")

# winter surface rise at the stations (daily medians, ±1 day; boom / stake distances shrink when the surface rises)
rise = {}
for st, (f, cols) in AWS_SENSORS.items():
    df = pd.read_csv(aws_dir / f, usecols=["time"] + cols, parse_dates=["time"]).set_index("time").resample("1D").median()

    def at(c, t):
        return df[c][t - pd.Timedelta("1D"):t + pd.Timedelta("1D")].median()

    ch = {c: (at(c, T_PROBE.normalize()) - at(c, T_END.normalize())) * (1 if c == "z_surf_combined" else -1) for c in cols}
    rise[st] = {"central": ch[cols[0]], "low": min(ch.values()), "high": max(ch.values())}
    print(f"{st}: " + ", ".join(f"{c} {v:+.3f}" for c, v in ch.items()) + " m")
w_u = (ELEV["site"] - ELEV["KAN_M"]) / (ELEV["KAN_U"] - ELEV["KAN_M"])
acc = {s: w_u * rise["KAN_U"][s] + (1 - w_u) * rise["KAN_M"][s] for s in ("central", "low", "high")}
print(f"site (weight KAN_U {w_u:.2f}): winter rise {acc['central']:.3f} m ({acc['low']:.3f}–{acc['high']:.3f})")

# GCP4 lowering 07-01 -> 09-20 of the central GCP model (b02 envelope)
e4 = env[env.gcp_id == 4].set_index("timestamp")
lower = {"central": (e4.y_central[T_END] - e4.y_central[T0]) / K_CAL,
         "low": e4.dy_p16[T_END] / K_CAL, "high": e4.dy_p84[T_END] / K_CAL}

# %%
# surfaces at the GCP4 pole, m below the pole top
end = {s: cal.emergence_m + lower[s] + melt_post for s in lower}
s26 = {"central": end["central"] - acc["central"], "low": end["low"] - acc["high"], "high": end["high"] - acc["low"]}
dep = pr.depth_cm.values / 100
dep4 = dep[np.argmin(abs(pr.distance_m.values - D_GCP4))]
chain = pd.DataFrame([
    ("2025-04-26 installation", EMERG_0426, np.nan, np.nan, "field emergence"),
    ("2025-07-01", cal.emergence_m, np.nan, np.nan, "emergence calibration"),
    ("end of 2025 melt", end["central"], end["low"], end["high"],
     f"+ GCP-model lowering {lower['central']:.3f} m (±1σ), + {melt_post * 100:.1f} cm after 09-20"),
    ("2026-04-23", s26["central"], s26["low"], s26["high"], f"− winter rise {acc['central']:.3f} m"),
    ("2026-04-23 probed ice", s26["central"] + dep4, np.nan, np.nan, f"+ probed depth {dep4:.2f} m at {D_GCP4:.0f} m"),
], columns=["surface", "below_top_m", "low_m", "high_m", "how"])
print(f"with the air-temperature estimate ({L_TEMP} m): end of melt {cal.emergence_m + L_TEMP + melt_post:.3f} m, "
      f"2026-04-23 {cal.emergence_m + L_TEMP + melt_post - acc['central']:.3f} m below the top")
print(f"probed ice below the end of 2025 melt: {dep4 - acc['central']:+.2f} m at GCP4, "
      f"mean {np.nanmean(dep) - acc['central']:+.2f} m ({np.nanmin(dep) - acc['central']:+.2f} to {np.nanmax(dep) - acc['central']:+.2f})")
chain.round(3)

# %%
gm07 = read_gcps(cam_dir / f"{cam}_gcps_img_{SURF_FRAME}.csv")           # manual picks (GCP4 = calibration base)
d = pr.distance_m.values
x = to_img07(on_line(d))[:, 0]
y_model = surface_line(gi07, x)
y_manual = surface_line(gm07, x)
s_ref = scale07(*gi07[3])
v = np.array([K_CAL * scale07(xi, yi) / s_ref for xi, yi in zip(x, y_model)])


def below(m):
    # image y of a surface m below the pole top (uniform offset from the manual 07-01 line)
    return y_manual + (m - cal.emergence_m) * v


y_end, y_26 = below(end["central"]), below(s26["central"])
y_ice = y_26 + dep * v
LINES = [("2025-04-26 installation: field emergence 0.72 m", below(EMERG_0426), dict(c="darkorange", lw=1.6)),
         (f"2025-04-26 installation: AWS estimate ({SPRING_AWS:.2f} m above 07-01)", below(cal.emergence_m - SPRING_AWS),
          dict(c="darkorange", lw=1.2, ls="--")),
         (f"2025-07-01 manual picks (GCP4 emergence {cal.emergence_m:.2f} m)", y_manual, dict(c="purple", lw=1.4, ls="--")),
         ("2025-07-01 modelled GCP4–GCP3 line (b02 central)", y_model, dict(c="magenta", lw=1.6)),
         (f"end of 2025 melt: 07-01 − GCP-model lowering {lower['central']:.3f} m", y_end, dict(c="crimson", lw=1.6)),
         (f"2026-04-23: end of melt + winter rise {acc['central']:.2f} m (KAN_M / KAN_U)", y_26, dict(c="royalblue", lw=1.8)),
         ("2026-04-23 probed ice: 2026-04-23 line − probed depth", y_ice, dict(c="teal", lw=1.4))]
POLE = sorted([(cal.y_top, "top")] + [(cal.y_top + m * K_CAL, f"{m:.2f} m ({lab})") for lab, m in [
    ("2025-04-26", EMERG_0426), ("2025-07-01", cal.emergence_m), ("2026-04-23", s26["central"]),
    ("end of 2025 melt", end["central"]), ("2026-04-23 probed ice", s26["central"] + dep4)]])
y_low = np.fmax(y_end, y_ice)                        # lowest line, for the tape labels

for lo, hi, zoom in [(0, 64, False), (6, 30, True)]:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(img)
    sel = (d >= lo) & (d <= hi)
    for lab, y, kw in LINES:
        ax.plot(x[sel], y[sel], label=lab, **kw)
    ax.plot([cal.x, cal.x], [cal.y_top, POLE[-1][0]], c="k", lw=1)
    for yy, t in POLE:
        ax.plot([cal.x - 6, cal.x + 6], [yy, yy], c="k", lw=1)
        if zoom:
            ax.annotate(t, (cal.x - 8, yy), ha="right", va="center", fontsize=7,
                        bbox=dict(fc="w", ec="none", alpha=0.7, pad=0.5))
    for lab, t in MARKS:
        if lo <= t <= hi:
            j = np.argmin(abs(d - t))
            ax.annotate(f"{t} m", (x[j], y_low[j]), (0, -12), textcoords="offset points", ha="center", va="top", fontsize=8)
    for k, nm in [(3, "GCP4"), (2, "GCP3")]:
        ax.annotate(nm, gi07[k], (8, -26), textcoords="offset points", fontsize=9)
    if zoom:
        ys = np.r_[y_low[sel], below(EMERG_0426)[sel]]
        ax.set_xlim(x[sel].min() - 60, x[sel].max() + 80)
        ax.set_ylim(np.nanmax(ys) + 80, np.nanmin(ys) - 160)
    else:
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    ax.set_title(f"{cam} {SURF_FRAME}, left stake line (GCP4 → GCP3): surfaces")
    ax.axis("off")
    fig.tight_layout()
    save_fig(fig, "surfaces_oblique", f"surfaces_{'zoom' if zoom else 'overview'}")
    plt.show()

# %% [markdown]
# ## 3. Channel widths (a07) in the oblique view
#
# The a07 cross-sections (`bathy_dir/<cam>/<cam>_xs_coord_<date>_<time>.csv`, UTM end points, all on one line parallel
# to the GCP1 → GCP2 baseline), projected into the oblique view with the GCPs of their own frame and drawn on the
# 2025-07-01 12:00 image. Orthoimage and back-projection use the same homography, so each line sits where its banks
# were picked; later lines lie lower because the modelled surface lowers.
#
# **Note:** the orthoimages and the cross-section picks still use the current GCP model (`_modelling/output`, OLS:
# the blue line in the vertical plots). The GCP model has since been improved (b02 central model); the
# orthorectification and the picks are to be redone.

# %%
from river.config import bathy_dir
import matplotlib.dates as mdates
from matplotlib import colors

XS_GCPS = mod_dir / "output"          # GCP model of the orthoimages and picks (current OLS model)


def xs_gcps(stamp):
    # same timestamp, else the first file of that date (as in a07)
    f = XS_GCPS / f"{cam}_gcps_img_{stamp}.csv"
    if not f.exists():
        f = sorted(XS_GCPS.glob(f"{cam}_gcps_img_{stamp[:8]}_*.csv"))[0]
    return read_gcps(f)


xs_rows = []
for f in sorted((bathy_dir / cam).glob(f"{cam}_xs_coord_*.csv")):
    stamp = f.stem.split("_xs_coord_")[1]
    p = pd.read_csv(f, header=None).values[:2, :2].astype(float)
    to_img, _ = homography(xs_gcps(stamp))
    uv = to_img(p)
    s, t = to_st(p)
    xs_rows.append({"time": pd.to_datetime(stamp, format="%Y%m%d_%H%M%S"), "width_m": np.linalg.norm(p[1] - p[0]),
                    "s1": s[0], "s2": s[1], "t": t.mean(), "x1": uv[0, 0], "y1": uv[0, 1], "x2": uv[1, 0], "y2": uv[1, 1]})
xs = pd.DataFrame(xs_rows)
print(f"{len(xs)} cross-sections {xs.time.min():%m-%d} -> {xs.time.max():%m-%d}, t {xs.t.min():.2f}–{xs.t.max():.2f} m, "
      f"width {xs.width_m.max():.2f} -> {xs.width_m.iloc[-1]:.2f} m")
xs.set_index("time").round(2)

# %%
img07 = frame_image(SURF_FRAME)
g07_xs = xs_gcps(SURF_FRAME)
tn = mdates.date2num(xs.time)
norm = colors.Normalize(tn.min(), tn.max())
cmap = plt.get_cmap("viridis")
fig, ax = plt.subplots(figsize=(13, 7.5))
ax.imshow(img07)
for r, c in zip(xs.itertuples(), cmap(norm(tn))):
    ax.plot([r.x1, r.x2], [r.y1, r.y2], "-", c=c, lw=1.6, alpha=0.6)
for k, nm in [(3, "GCP4"), (2, "GCP3"), (0, "GCP1")]:
    ax.plot(*g07_xs[k], "o", mfc="none", mec="magenta", ms=9, mew=1.2)
    ax.annotate(nm, g07_xs[k], (8, -10), textcoords="offset points", color="magenta", fontsize=9)
ax.set_xlim(xs[["x1", "x2"]].values.min() - 250, max(xs[["x1", "x2"]].values.max(), g07_xs[3, 0]) + 250)
ax.set_ylim(xs[["y1", "y2"]].values.max() + 150, xs[["y1", "y2"]].values.min() - 200)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.03, pad=0.01)
cb.ax.yaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
cb.set_label("cross-section (a07 pick), 2025")
ax.text(0.01, 0.02, "Note: orthoimages and cross-section picks still use the current GCP model\n"
        "(_modelling/output, OLS = blue line in the vertical plots); the GCP model has since been\n"
        "improved (b02 central model). Orthorectification and picks to be redone.",
        transform=ax.transAxes, fontsize=8, va="bottom", bbox=dict(fc="w", ec="0.6", alpha=0.85))
ax.set_title(f"{cam}: a07 channel widths {xs.time.min():%m-%d} – {xs.time.max():%m-%d} (each with its frame's GCPs) on {SURF_FRAME}")
ax.axis("off")
fig.tight_layout()
save_fig(fig, "channel_width_oblique", "channel_widths_oblique")
plt.show()


# %% [markdown]
# ## 4. Channel at the modelled surface and at the water surface
#
# The a07 bank points are the bank tops on the modelled surface (current model). The bank height (a05 water surface
# (bank), same date, frame closest to 12:00) is the drop from there down to the water surface. With steep banks the
# water edges lie directly below the bank tops: same width, one bank height lower (in the image: bank height ×
# vertical scale lower).

# %%
vm_files = sorted((cam_dir / "vert_meas").glob(f"{cam}_vert_meas_*.csv"))
WS_SHOW = ["20250723", "20250820"]                  # dates for the oblique panels


def vm_hour(f):
    return int(f.stem[-6:-4])


ws_rows, ws_img = [], {}
for r in xs.itertuples():
    date = f"{r.time:%Y%m%d}"
    cands = [f for f in vm_files if f.stem.split("_vert_meas_")[1][:8] == date]
    if not cands:
        continue
    f = sorted(cands, key=lambda f: (abs(vm_hour(f) - 12), -vm_hour(f)))[0]
    vm = pd.read_csv(f)
    ws = vm[vm.kind == "water_surface"]
    if ws.empty:
        continue
    gi = xs_gcps(f"{r.time:%Y%m%d_%H%M%S}")
    _, scale = homography(gi)
    s_gcp4 = scale(*gi[3])
    xm, yp = float(ws.x.iloc[0]), float(ws.y_point.iloc[0])
    y_s = float(surface_line(gi, xm))
    dz = (yp - y_s) / (K_CAL * scale(xm, y_s) / s_gcp4)          # bank height below the current-model surface
    top = np.array([[r.x1, r.y1], [r.x2, r.y2]])
    down = np.array([[x, y + dz * K_CAL * scale(x, y) / s_gcp4] for x, y in top])
    ws_rows.append({"date": r.time.normalize(), "a05_frame": f.stem[-15:], "bank_height_m": dz, "width_m": r.width_m,
                    "s1": min(r.s1, r.s2), "s2": max(r.s1, r.s2)})
    ws_img[date] = (r.time, top, down, ws_rows[-1])
ws_df = pd.DataFrame(ws_rows)
ws_df.round(3)

# %%
fig = plt.figure(figsize=(14, 10.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1])
for k, date in enumerate(WS_SHOW):
    t, top, down, r = ws_img[date]
    ax = fig.add_subplot(gs[0, k])
    ax.imshow(frame_image(f"{t:%Y%m%d_%H%M%S}"))
    ax.plot(top[:, 0], top[:, 1], "-", c="magenta", lw=1.8, label=f"a07 bank tops on the modelled surface: {r['width_m']:.2f} m")
    ax.plot(down[:, 0], down[:, 1], "-", c="deepskyblue", lw=1.8,
            label=f"water surface, {r['bank_height_m']:.2f} m below (steep banks: same {r['width_m']:.2f} m)")
    for (x0, y0), (x1, y1) in zip(top, down):
        ax.plot([x0, x1], [y0, y1], "-", c="k", lw=1)
    pts = np.r_[top, down]
    ax.set_xlim(pts[:, 0].min() - 200, pts[:, 0].max() + 200)
    ax.set_ylim(pts[:, 1].max() + 150, pts[:, 1].min() - 150)
    ax.set_title(f"{t:%Y-%m-%d %H:%M} (bank height from a05 {r['a05_frame']})", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.axis("off")
ax = fig.add_subplot(gs[1, :])
for (_, r), c in zip(ws_df.iterrows(), plt.get_cmap("viridis")(np.linspace(0, 1, len(ws_df)))):
    ax.plot([r.s1, r.s1, r.s2, r.s2], [0, -r.bank_height_m, -r.bank_height_m, 0], "-", c=c, lw=1.8,
            label=f"{r.date:%m-%d}: {r.width_m:.2f} m, bank {r.bank_height_m:.2f} m")
ax.axhline(0, c="magenta", lw=1, ls="--", label="modelled surface (current model)")
ax.set_xlabel("s along GCP1 → GCP2 (m), a07 cross-section line")
ax.set_ylabel("level relative to the modelled surface (m)")
ax.set_title("Channel at the modelled surface (bank tops) and at the water surface (steep banks assumed)", fontsize=10)
ax.legend(loc="lower right", fontsize=7, ncol=2)
ax.grid(alpha=0.3)
ax.set_ylim(-0.75, 0.15)
fig.text(0.01, 0.005, "Orthoimages, picks and bank heights use the current GCP model (_modelling/output, OLS); "
         "to be redone with the improved model.", fontsize=8)
fig.tight_layout(rect=(0, 0.02, 1, 1))
save_fig(fig, "channel_width_oblique", "channel_width_water_surface")
plt.show()


# %% [markdown]
# ## 5. Cross-sections at the water level (a07a_xs_frb)
#
# Selections of `a07a_xs_frb.py` (`data/xs_frb/<cam>/<model>/`): far and close bank on the fixed line (modelled
# surface, vertical banks), water surface at the far bank and the same freeboard at the close bank. The cross-section
# at the water level is the line between the two water points in the image; the camera is fixed, so it can be drawn
# on any frame. With the probed bathymetry: the ice surface of 2026-04-23 from section 2 (modelled 2026-04-23
# surface − probed depth), i.e. the stream bed at the end of the 2025 melt. It was probed along the left stake line,
# about 1 m beside the cross-section line; the bathymetry is assumed constant along the river, so each probe is moved
# by its local x onto the cross-section line, at its height below the 2025-07-01 surface. Overview on the 2025-07-01
# image; one figure per selection on its own frame (saved in `xs_water_level/per_date`, not shown); the cross-section
# with the water levels and the bathymetry. Sections 3 and 4 keep the a07 picks and the a05 bank heights.

# %%
XS_FRB_MODEL = "central"                            # GCP model of the a07a selections
XS_LINE_Y = 9.2406                                  # local y of the fixed cross-section line (a07 / a07a)
frb_files = sorted((data_dir / "xs_frb" / cam / XS_FRB_MODEL).glob(f"{cam}_xs_frb_*.csv"))


def water_line(r):
    # image points of the water line at the far and the close bank (also the first test files)
    if "img_x1" in r and pd.notna(r["img_x1"]):
        return (r["img_x1"], r["img_y_water1"]), (r["img_x2"], r["img_y_water2"])
    return (r["x1"], r["y_water1"]), (r["x2"], r["y_water2"])


frb_rows = []
for f in frb_files:
    r = pd.read_csv(f).iloc[0]
    d_, t_ = f.stem.split("_")[-2:]
    (xa, ya), (xb, yb) = water_line(r)
    new = "local_x1" in r
    frb_rows.append({"time": pd.to_datetime(d_ + t_, format="%Y%m%d%H%M%S"), "stamp": f"{d_}_{t_}",
                     "width_m": r["width_m"], "freeboard_m": r["freeboard_m"], "xa": xa, "ya": ya, "xb": xb, "yb": yb,
                     "lx1": r["local_x1" if new else "s1"], "lx2": r["local_x2" if new else "s2"],
                     "y_top1": r["img_y1" if new else "y1"]})
frb = pd.DataFrame(frb_rows).sort_values("time").reset_index(drop=True)

# probed bathymetry (section 2) moved onto the cross-section line by local x, constant along the river:
# height below the 2025-07-01 surface = 2026-04-23 surface below it + probed depth
probe_lx = pr["s_local"].values                             # local x of the probes (left stake line)
probe_drop = (s26["central"] - cal.emergence_m) + dep        # m below the 2025-07-01 surface


def xs_surface_07(lx):
    # image point of the 2025-07-01 surface (central model) on the cross-section line at local x, and px per m there
    pts = to_img07(G[1] + np.outer(np.atleast_1d(lx), u12) + XS_LINE_Y * n12)
    vpx = np.array([K_CAL * scale07(a, b) / s_ref for a, b in pts])
    return pts[:, 0], pts[:, 1], vpx


h07, w07 = img07.shape[:2]
px0, py0, vp0 = xs_surface_07(probe_lx)
probe_x, probe_y = px0, py0 + probe_drop * vp0
inside = np.isfinite(probe_y) & (probe_x >= 0) & (probe_x <= w07) & (probe_y >= 0) & (probe_y <= h07)
probe_x, probe_y = np.where(inside, probe_x, np.nan), np.where(inside, probe_y, np.nan)

# one view for all figures: the water lines and the probed bathymetry
px = np.r_[frb["xa"], frb["xb"], probe_x[inside]]
py = np.r_[frb["ya"], frb["yb"], probe_y[inside]]
VIEW_WL = ((max(px.min() - 150, 0), min(px.max() + 150, w07)), (min(py.max() + 120, h07), max(py.min() - 150, 0)))
print(f"{len(frb)} selections ({XS_FRB_MODEL}): {frb['time'].min():%m-%d %H:%M} -> {frb['time'].max():%m-%d %H:%M}")
frb[["time", "width_m", "freeboard_m"]].round(3)

# %%
tn_wl = mdates.date2num(frb["time"])
norm_wl = colors.Normalize(tn_wl.min(), tn_wl.max())
cmap_wl = plt.get_cmap("viridis")
fig, ax = plt.subplots(figsize=(13, 7.5))
ax.imshow(img07)
for r, c in zip(frb.itertuples(), cmap_wl(norm_wl(tn_wl))):
    ax.plot([r.xa, r.xb], [r.ya, r.yb], "-", c=c, lw=1.8, alpha=0.85)
ax.plot(probe_x, probe_y, "-", c="orangered", lw=1.8, label="probed bathymetry (end of the 2025 melt)")
ax.set_xlim(VIEW_WL[0])
ax.set_ylim(VIEW_WL[1])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm_wl, cmap=cmap_wl), ax=ax, fraction=0.03, pad=0.01)
cb.ax.yaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
cb.set_label("cross-section at the water level, 2025")
ax.legend(loc="lower right", fontsize=9)
ax.set_title(f"{cam}: cross-sections at the water level (a07a, {XS_FRB_MODEL} model) and the probed bathymetry, "
             f"on {SURF_FRAME}", fontsize=11)
ax.axis("off")
fig.tight_layout()
save_fig(fig, "xs_water_level", f"xs_water_level_oblique_{XS_FRB_MODEL}")
plt.show()

# %%
# one figure per selection on its own frame (saved, not shown)
for r in frb.itertuples():
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.imshow(frame_image(r.stamp))
    ax.plot([r.xa, r.xb], [r.ya, r.yb], "-", c="deepskyblue", lw=2,
            label=f"water level {r.time:%Y-%m-%d %H:%M}: width {r.width_m:.2f} m, freeboard {r.freeboard_m:.2f} m")
    ax.plot(probe_x, probe_y, "-", c="orangered", lw=1.8, label="probed bathymetry (end of the 2025 melt)")
    ax.set_xlim(VIEW_WL[0])
    ax.set_ylim(VIEW_WL[1])
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"{cam} {r.time:%Y-%m-%d %H:%M} ({XS_FRB_MODEL} model)", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    save_fig(fig, "xs_water_level", f"per_date/xs_water_level_{XS_FRB_MODEL}_{r.stamp}")
    plt.close(fig)
print(f"{len(frb)} figures in {FIG_ROOT / 'xs_water_level' / 'per_date'}")

# %%
# cross-section: water levels and probed bathymetry, height relative to the 2025-07-01 surface (central model);
# water level = bank top of the far bank (modelled surface of that frame, from its image position) − freeboard
wl_rows = []
for r in frb.itertuples():
    _, y0, v0 = xs_surface_07(r.lx1)
    h_top = -(r.y_top1 - y0[0]) / v0[0]                         # modelled surface at the far bank vs 2025-07-01 (m)
    wl_rows.append({"time": r.time, "lx1": r.lx1, "lx2": r.lx2, "level_m": h_top - r.freeboard_m})
wl = pd.DataFrame(wl_rows)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(probe_lx, -probe_drop, "-", c="orangered", lw=1.8, label="probed bathymetry (end of the 2025 melt)")
for r, c in zip(wl.itertuples(), cmap_wl(norm_wl(tn_wl))):
    ax.plot([r.lx2, r.lx1], [r.level_m, r.level_m], "-", c=c, lw=1.6)
ax.axhline(0, c="k", lw=0.8, ls="--", label="modelled surface 2025-07-01")
ax.axhline(cal.emergence_m - end["central"], c="crimson", lw=0.8, ls=":", label="modelled surface, end of the 2025 melt")
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm_wl, cmap=cmap_wl), ax=ax, fraction=0.04, pad=0.01)
cb.ax.yaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
cb.set_label("water level, 2025")
ax.set_xlim(wl["lx2"].min() - 2, wl["lx1"].max() + 3)
ax.set_xlabel("local x along the cross-section (m), camera side on the left")
ax.set_ylabel("height relative to the 2025-07-01 surface (m)")
ax.set_title(f"{cam}: water levels (a07a, {XS_FRB_MODEL} model) and the probed bathymetry", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
save_fig(fig, "xs_water_level", f"xs_water_level_profile_{XS_FRB_MODEL}")
plt.show()
wl.round(3)
