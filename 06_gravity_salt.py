import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from tqdm import tqdm

# =========================================================================
# CONFIGURATION PARAMETERS - EDIT HERE
# =========================================================================

# Input/Output settings
img_path = "dataset/salt/salt_basement.png"  # Path to input image
save_folder = "salt"                          # Output folder
save_result = "gz_profile.png"  # Output filename

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
# 3) Single-station, explicit 2D for-loop (reference/clarity)
#    g_z(x') = sum_{z=0..H-1} sum_{x=0..W-1} G * z / [(x'-x)^2 + z^2]^(3/2) * σ[x,z]
#    Downward is positive; z is the pixel row index (0 at top).
# ------------------------------------------------------------
def gz_single_station_loop(xg):
    acc = 0.0
    for z in tqdm(range(H), desc="Computing gravity (loop method)", leave=False):
        dz = float(z)  # since z' = 0 and pixel location is z (no +0.5)
        for x in range(W):
            dx = xg - float(x)
            r2 = dx*dx + dz*dz
            r3 = (r2 + eps)**1.5     # eps prevents division by zero at (dx=0, dz=0)
            acc += G * dz / r3 * sigma2d_padded[z, x]
    return acc  # [m/s^2]

# Example check: first station via loop
with tqdm(total=100, desc="Station 0 calculation") as pbar:
    gz0_ms2_loop = gz_single_station_loop(xg_pix[0])
    gz0_mgal_loop = gz0_ms2_loop * MS2_TO_MGAL
    pbar.update(100)

# ------------------------------------------------------------
# 4) Matrix form: g = A_z σ
#    Build A_z for all stations at once.
#    Mass coordinates: x_i in [0..W-1], z_i in [0..H-1], flattened row-major.
#    A_{j,i} = G * z_i / [ (x'_j - x_i)^2 + z_i^2 ]^(3/2)
# ------------------------------------------------------------
# Mass coordinates (flattened in 'C' order: row z runs slowest? Actually in C: last axis changes fastest → x varies fastest)
with tqdm(total=100, desc="Building coordinate arrays") as pbar:
    # [0,1,...,W-1, 0,1,...] length H*W
    x_i = np.tile(np.arange(W, dtype=float), H)         
    pbar.update(50)
    # [0,0,...,0, 1,1,...,1, ...] length H*W
    z_i = np.repeat(np.arange(H, dtype=float), W)       
    pbar.update(50)

# Broadcast station x' against all mass (x_i, z_i)
with tqdm(total=100, desc="Building gravity matrix") as pbar:
    dx = xg_pix[:, None] - x_i[None, :]                 # (M, N)
    pbar.update(25)
    dz = z_i[None, :]                                   # (1, N) since z' = 0
    pbar.update(25)
    r2 = dx*dx + dz*dz
    pbar.update(25)
    r3 = (r2 + eps)**1.5
    Az = G * dz / r3                                    # (M, N)
    pbar.update(25)

# Forward model
with tqdm(total=100, desc="Computing forward model") as pbar:
    gz_ms2 = Az @ sigma_vec                             # (M,)
    pbar.update(50)
    gz_mgal = gz_ms2 * MS2_TO_MGAL
    pbar.update(50)

# ------------------------------------------------------------
# 5) Plot
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

axes[0].imshow(sigma2d_padded, cmap="gray", origin="upper")
axes[0].scatter(xg_pix, np.zeros_like(xg_pix), s=14, c="red", marker="^", label="Stations z'=0")
# Add dashed lines to show synthetic expansion boundaries
axes[0].axvline(x=pad_width, color='blue', linestyle='--', alpha=0.7, label='Synthetic boundary')
axes[0].axvline(x=pad_width + original_width, color='blue', linestyle='--', alpha=0.7)
axes[0].set_title("σ-image (padded, each pixel = one point mass)")
axes[0].set_xlabel("x (px)")
axes[0].set_ylabel("z (px)")
axes[0].legend(loc="lower right")

station_numbers = np.arange(1, len(xg_pix) + 1)
axes[1].plot(station_numbers, gz_mgal, lw=1.8)
axes[1].set_title(r"Predicted $g_z$ along surface (downward $+$)")
axes[1].set_xlabel("Gravitational Station")
axes[1].set_ylabel(r"$g_z$ (mGal)")
plt.show()

fig.savefig(OUTDIR / save_result, dpi=300)
plt.close(fig)

