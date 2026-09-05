import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

from lib.control_plot import PLOT_PARAMS  # shared global plot style
from lib.util import plot_gravity_matrix  # shared plotting (see lib/util.py)

matplotlib.rcParams.update(PLOT_PARAMS)

# ─────────────────────────────────────────────────────────────
# Output folder
# ─────────────────────────────────────────────────────────────
OUTDIR = Path("ch_02_02_gravity_matrix")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Constants & units
# ─────────────────────────────────────────────────────────────
G = 6.67430e-11   # m^3 kg^-1 s^-2
MS2_TO_MGAL = 1e5 # 1 m/s^2 = 1e5 mGal

# ─────────────────────────────────────────────────────────────
# Geometry: M surface stations, N source cells (no y-dimension)
# ─────────────────────────────────────────────────────────────
x_min, x_max, dx = -200.0, 200.0, 1.0
x_obs = np.arange(x_min, x_max + dx, dx)  # (M,)
M = x_obs.size

# Five source cells (x_i, z_i) with depths positive downward
x_i = np.array([-15.0, -5.0, 0.0, 7.0, 10.0], dtype=float)   # (N,)
z_i = np.array([ 40.0, 60.0, 80.0, 60.0, 40.0], dtype=float) # (N,)
N = x_i.size

# Cell areas ΔA_i (m^2). Use the true cell areas if you have them.
# For demonstration, set all to 1.0 m^2.
dA_i = np.full(N, 1.0, dtype=float)

# OPTION A: Give surface densities σ_i directly (kg/m^2)
sigma_i = None  # e.g., np.array([...], dtype=float)

# OPTION B: Start from masses m_i (kg) and convert via σ_i = m_i / ΔA_i
m_i = np.array([2.0e9, 1.5e9, 3.0e9, 1.0e9, 2.5e9], dtype=float)  # (N,)
if sigma_i is None:
    sigma_i = m_i / dA_i  # kg/m^2

# ─────────────────────────────────────────────────────────────
# Design matrix  A_z  and forward model  g_z = A_z @ sigma
# ─────────────────────────────────────────────────────────────
def build_Az(x_obs: np.ndarray,
             x_i: np.ndarray,
             z_i: np.ndarray,
             dA_i: np.ndarray) -> np.ndarray:
    """
    Assemble A_z of shape (M, N) with entries:
        A_{ji} = G * ΔA_i * z_i / r_{ji}^3,
    where r_{ji}^2 = (x'_j - x_i)^2 + z_i^2  (no y-dimension).
    Units: A in (m/s^2) per (kg/m^2) = m^3/(kg·s^2) * m^2 / m^3 = m/s^2 per (kg/m^2).
    """
    # Broadcast to MxN
    dx = x_obs[:, None] - x_i[None, :]        # (M, N)
    r2 = dx**2 + (z_i[None, :]**2)            # (M, N)
    r3 = r2**1.5                               # (M, N)
    Az = G * (dA_i[None, :] * z_i[None, :]) / r3
    return Az

A_z = build_Az(x_obs, x_i, z_i, dA_i)  # (M, N)

# Per-source contributions (M, N) and total (M,)
gz_each_ms2 = A_z * sigma_i[None, :]          # elementwise multiply
gz_total_ms2 = gz_each_ms2.sum(axis=1)

# Convert to mGal for plotting
gz_each_mgal = gz_each_ms2 * MS2_TO_MGAL
gz_total_mgal = gz_total_ms2 * MS2_TO_MGAL

# ─────────────────────────────────────────────────────────────
# Plotting (shared implementation in lib/util.py)
# ─────────────────────────────────────────────────────────────
plot_gravity_matrix(gz_each_mgal, gz_total_mgal, x_obs, x_i, z_i, OUTDIR, x_min, x_max, dpi=300)