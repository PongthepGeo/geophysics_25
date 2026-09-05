import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

from lib.control_plot import PLOT_PARAMS  # shared global plot style
from lib.progress import Progress, StageTimer  # shared progress reporting (see lib/progress.py)

matplotlib.rcParams.update(PLOT_PARAMS)

# =========================================================================
# CONFIGURATION PARAMETERS - EDIT HERE
# =========================================================================

# Input/Output settings
img_path = "dataset/salt/salt_basement.png"  # Path to input image
save_folder = "ch_02_03_gravity_salt"         # Output folder
save_sigma_image = "sigma_image.png"  # Output filename (density-contrast map)
save_result = "gz_profile.png"        # Output filename (predicted gz profile)

# Density contrast values for different pixel intensities
SALT_PIXEL_VALUE = 94      # Pixel value representing salt bodies
BASEMENT_PIXEL_VALUE = 76  # Pixel value representing basement
SALT_DENSITY = 2.2         # Density contrast for salt (g/cm³)
BASEMENT_DENSITY = 3.1     # Density contrast for basement (g/cm³)
# Background/other pixels = 0 (no density contrast)

# Gravity survey parameters
PAD_PERCENTAGE = 0.1       # Padding percentage (10% on each side)
AVOID_EDGE = 10           # Pixels to avoid at image edges
NUM_STATIONS = 50         # Number of gravity measurement stations

# Physical constants
G = 6.67430e-11           # Gravitational constant (m³ kg⁻¹ s⁻²)
MS2_TO_MGAL = 1e5         # Conversion factor: 1 m/s² = 1e5 mGal
eps = 1e-12               # Small number to prevent division by zero

# Demo pacing: the real computation below is fast enough to finish before a
# class can read the progress bar. These add an artificial delay so each
# stage "breathes" at a visible pace -- set both to 0 to run at full speed.
STATION_PACE_SEC = 0.0   # sleep per station in the reference loop (real timing is enough)
STAGE_PACE_SEC = 1.5     # sleep added to each vectorized (StageTimer) stage

# =========================================================================
# END CONFIGURATION
# =========================================================================

OUTDIR = Path(save_folder); OUTDIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# 1) Read image as σ-map
#    Each pixel is one point mass (area = 1 m^2).
#    No scaling/conversion from grayscale.
# -------------------------
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
# Quick check of image
plt.imshow(img, cmap="gray", origin="upper")
plt.show()
if img is None:
    raise FileNotFoundError(f"Could not read {img_path}")
H, W = img.shape
# Convert pixels to density contrasts based on configuration
salt = np.where(img == SALT_PIXEL_VALUE, SALT_DENSITY, 0.0)
basement = np.where(img == BASEMENT_PIXEL_VALUE, BASEMENT_DENSITY, 0.0)
sigma2d = salt + basement  # Combine salt and basement, background pixels remain 0

# Pad image left and right with zeros based on configuration
pad_width = int(W * PAD_PERCENTAGE)
sigma2d_padded = np.pad(sigma2d, ((0, 0), (pad_width, pad_width)), mode='constant', constant_values=0)
H, W_padded = sigma2d_padded.shape
original_width = W
W = W_padded
sigma_vec = sigma2d_padded.ravel(order="C")      # length N = H*W

# -------------------------
# 2) Gravimeter stations (z' = 0), x' along the top edge
#    No pixel-center offset; x, z are integer grid coordinates.
# -------------------------
xg_pix = np.linspace(AVOID_EDGE, W-AVOID_EDGE, NUM_STATIONS, dtype=float)  # station x' in pixel units
zg = 0.0  # all stations at the surface (z' = 0)


# ------------------------------------------------------------
# 3) Per-station, explicit 2D for-loop (reference/clarity)
#    g_z(x') = sum_{z=0..H-1} sum_{x=0..W-1} G * z / [(x'-x)^2 + z^2]^(3/2) * σ[x,z]
#    Downward is positive; z is the pixel row index (0 at top).
# ------------------------------------------------------------
def gz_single_station_loop(xg):
    acc = 0.0
    for z in range(H):
        dz = float(z)  # since z' = 0 and pixel location is z (no +0.5)
        for x in range(W):
            dx = xg - float(x)
            r2 = dx*dx + dz*dz
            r3 = (r2 + eps)**1.5     # eps prevents division by zero at (dx=0, dz=0)
            acc += G * dz / r3 * sigma2d_padded[z, x]
    return acc  # [m/s^2]

# Reference check: every station via the brute-force loop above, so it can
# be compared against the matrix-form result computed in section 4.
gz_loop_ms2 = np.empty(NUM_STATIONS, dtype=float)
with Progress(total=NUM_STATIONS, label="gz stations", unit="station") as bar:
    for s, xg in enumerate(xg_pix):
        gz_loop_ms2[s] = gz_single_station_loop(xg)
        time.sleep(STATION_PACE_SEC)
        bar.update(1)
gz_loop_mgal = gz_loop_ms2 * MS2_TO_MGAL

# ------------------------------------------------------------
# 4) Matrix form: g = A_z σ
#    Build A_z for all stations at once.
#    Mass coordinates: x_i in [0..W-1], z_i in [0..H-1], flattened row-major.
#    A_{j,i} = G * z_i / [ (x'_j - x_i)^2 + z_i^2 ]^(3/2)
# ------------------------------------------------------------
# Mass coordinates (flattened in 'C' order: row z runs slowest? Actually in C: last axis changes fastest → x varies fastest)
with StageTimer("coord arrays"):
    # [0,1,...,W-1, 0,1,...] length H*W
    x_i = np.tile(np.arange(W, dtype=float), H)
    # [0,0,...,0, 1,1,...,1, ...] length H*W
    z_i = np.repeat(np.arange(H, dtype=float), W)
    time.sleep(STAGE_PACE_SEC)

# Broadcast station x' against all mass (x_i, z_i)
with StageTimer("gravity matrix"):
    dx = xg_pix[:, None] - x_i[None, :]                 # (M, N)
    dz = z_i[None, :]                                   # (1, N) since z' = 0
    r2 = dx*dx + dz*dz
    r3 = (r2 + eps)**1.5
    Az = G * dz / r3                                    # (M, N)
    time.sleep(STAGE_PACE_SEC)

# Forward model
with StageTimer("forward model"):
    gz_ms2 = Az @ sigma_vec                             # (M,)
    gz_mgal = gz_ms2 * MS2_TO_MGAL
    time.sleep(STAGE_PACE_SEC)

# Sanity check: loop (section 3) vs matrix (section 4) should agree
max_diff_mgal = np.max(np.abs(gz_loop_mgal - gz_mgal))
print(f"Loop vs matrix max difference: {max_diff_mgal:.3e} mGal")

# ------------------------------------------------------------
# 5) Plot -- two separate figures, saved separately
# ------------------------------------------------------------

# 5a) sigma-image: density-contrast map with stations + padding boundaries
fig_sigma, ax_sigma = plt.subplots()
ax_sigma.imshow(sigma2d_padded, cmap="gray", origin="upper")
ax_sigma.scatter(xg_pix, np.zeros_like(xg_pix), s=14, c="red", marker="^", label="Stations z'=0")
# Add dashed lines to show synthetic expansion boundaries
ax_sigma.axvline(x=pad_width, color='blue', linestyle='--', alpha=0.7, label='Synthetic boundary')
ax_sigma.axvline(x=pad_width + original_width, color='blue', linestyle='--', alpha=0.7)
ax_sigma.set_title("σ-image (padded, each pixel = one point mass)")
ax_sigma.set_xlabel("x (px)")
ax_sigma.set_ylabel("z (px)")
ax_sigma.legend(loc="lower right")
plt.show()

sigma_fig_path = OUTDIR / save_sigma_image
fig_sigma.savefig(sigma_fig_path, dpi=300)
plt.close(fig_sigma)

# 5b) predicted gz profile
fig_profile, ax_profile = plt.subplots()
station_numbers = np.arange(1, len(xg_pix) + 1)
ax_profile.plot(station_numbers, gz_mgal, lw=1.8)
ax_profile.set_title(r"Predicted $g_z$ along surface (downward $+$)")
ax_profile.set_xlabel("Gravitational Station")
ax_profile.set_ylabel(r"$g_z$ (mGal)")
plt.show()

profile_fig_path = OUTDIR / save_result
fig_profile.savefig(profile_fig_path, dpi=300)
plt.close(fig_profile)

print("Saved figures:")
print(" -", sigma_fig_path)
print(" -", profile_fig_path)

