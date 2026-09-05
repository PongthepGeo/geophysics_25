import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Output folder
# ─────────────────────────────────────────────────────────────
OUTDIR = Path("halfspace_gz_demo")
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
# Plotting
# ─────────────────────────────────────────────────────────────
def plot_gravity_matrix(gz_each_mgal: np.ndarray,
                        gz_total_mgal: np.ndarray,
                        x_obs: np.ndarray,
                        x_i: np.ndarray,
                        z_i: np.ndarray,
                        OUTDIR: Path,
                        x_min: float,
                        x_max: float) -> None:
    # 1) Profile: all sources + total
    plt.figure(figsize=(10, 5))
    N = gz_each_mgal.shape[1]
    for sidx in range(N):
        plt.plot(x_obs, gz_each_mgal[:, sidx], linewidth=1.8, label=f"source {sidx+1}")
    plt.plot(x_obs, gz_total_mgal, linewidth=2.6, linestyle="-", label="total")

    for xi in x_i:
        plt.axvline(xi, linestyle="--", linewidth=1)

    plt.title(r"Vertical gravity $g_z$ along surface (matrix form: $g_z=A_z\,\sigma$)")
    plt.xlabel("x (m) at surface")
    plt.ylabel("g_z (mGal, downward +)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncols=3, frameon=True)
    fig_profile = OUTDIR / "gz_profile_all.png"
    plt.tight_layout()
    plt.savefig(fig_profile, format="png")
    plt.show()

    # 2) Geometry figure
    plt.figure(figsize=(10, 3.6))
    plt.hlines(0.0, x_min, x_max, linestyles="-", linewidth=2, label="surface (z=0)")
    gstep = 25
    gpos = x_obs[::gstep]
    plt.scatter(gpos, np.zeros_like(gpos), marker="^", s=20, label="gravimeters")
    plt.scatter(x_i, z_i, marker="o", s=50, label="source cells")
    for idx, (xi, zi) in enumerate(zip(x_i, z_i), start=1):
        plt.text(xi, zi, f"  i={idx}", va="center")

    plt.xlim(x_min, x_max)
    zmax = max(z_i.max() + 20.0, 60.0)
    plt.ylim(0.0, zmax)
    ax = plt.gca()
    ax.invert_yaxis()  # depth increases downward
    plt.xlabel("x (m)")
    plt.ylabel("z (m, downward +)")
    plt.title("Half-space geometry: source cells and surface gravimeters")
    plt.legend(frameon=True)
    plt.tight_layout()
    fig_geom = OUTDIR / "geometry_sources_gravimeters.png"
    plt.savefig(fig_geom, format="png")
    plt.show()

    print("Saved figures:")
    print(" -", fig_profile)
    print(" -", fig_geom)

# Run plots
plot_gravity_matrix(gz_each_mgal, gz_total_mgal, x_obs, x_i, z_i, OUTDIR, x_min, x_max)
