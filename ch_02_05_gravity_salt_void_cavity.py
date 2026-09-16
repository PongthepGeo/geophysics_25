import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

from lib.control_plot import PLOT_PARAMS  # shared global plot style
from lib.util import plot_gravity_potential_gradient  # shared plotting (see lib/util.py)

matplotlib.rcParams.update(PLOT_PARAMS)

# ─────────────────────────────────────────────────────────────
# Survey example (Section 5): a large evaporite-dissolution void
# left beneath a CAES site -- same equations as
# ch_02_04_gravity_matrix_potential_gradient.py, applied to a
# mass-deficit (void) instead of an excess mass.
# ─────────────────────────────────────────────────────────────
OUTDIR = Path("ch_02_05_gravity_salt_void_cavity")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Constants & units
# ─────────────────────────────────────────────────────────────
G = 6.67430e-11    # m^3 kg^-1 s^-2
MS2_TO_MGAL = 1e5  # 1 m/s^2 = 1e5 mGal
SI_TO_EOTVOS = 1e9 # 1 Eotvos (E) = 1e-9 s^-2

# ─────────────────────────────────────────────────────────────
# Geometry: one large void cavity, M surface stations (no y-dimension)
# ─────────────────────────────────────────────────────────────
x_min, x_max, dx = -1500.0, 1500.0, 5.0
x_obs = np.arange(x_min, x_max + dx, dx)  # (M,)
M = x_obs.size

# Void cavity: centre (x_i, z_i), nominal radius r_body, density contrast drho.
# Halite host (~2200 kg/m^3) leached out and left brine-filled --
# so the void is *less* dense than the host: drho < 0 (mass deficit).
x_i = np.array([0.0])     # (N=1,) cavity centred at x=0
z_i = np.array([400.0])   # (N=1,) depth to centre, m -- deep evaporite section
r_body = 150.0            # nominal cavity radius, m ("large" cavity)
drho = -1200.0            # density contrast, kg/m^3 (halite minus brine fill)

# Equivalent point mass: m = drho * (4/3) pi r^3  (mass DEFICIT, m < 0)
volume = (4.0 / 3.0) * np.pi * r_body**3       # sphere volume, m^3
m_i = np.array([drho * volume])                # (N=1,) kg, negative

# "Circle-like" outline for the geometry panel only: a natural dissolution
# cavity is not a perfect circle, so perturb the radius with a couple of
# smooth angular harmonics. The physics still uses the simple equivalent
# point-mass approximation above (r_body, drho) -- only the artwork here
# is irregular.
theta = np.linspace(0.0, 2.0 * np.pi, 72)
r_wobble = r_body * (1.0 + 0.15 * np.cos(3 * theta + 0.4) + 0.08 * np.sin(5 * theta + 1.1))
outline_x = x_i[0] + r_wobble * np.cos(theta)
outline_z = z_i[0] + r_wobble * np.sin(theta)
body_outline = np.column_stack([outline_x, outline_z])

# ─────────────────────────────────────────────────────────────
# Design matrices (same broadcasting pattern as ch_02_02_gravity_matrix.py
# / ch_02_04_gravity_matrix_potential_gradient.py)
# ─────────────────────────────────────────────────────────────
def build_Az(x_obs: np.ndarray, x_i: np.ndarray, z_i: np.ndarray) -> np.ndarray:
    """A_z[j,i] = G * z_i / r_ji^3, so g_z = A_z @ m  (m/s^2)."""
    dxx = x_obs[:, None] - x_i[None, :]   # (M, N)
    r2 = dxx**2 + z_i[None, :]**2
    r3 = r2**1.5
    return G * z_i[None, :] / r3

def build_AV(x_obs: np.ndarray, x_i: np.ndarray, z_i: np.ndarray) -> np.ndarray:
    """A_V[j,i] = -G / r_ji, so V = A_V @ m  (m^2/s^2, i.e. J/kg)."""
    dxx = x_obs[:, None] - x_i[None, :]   # (M, N)
    r = np.sqrt(dxx**2 + z_i[None, :]**2)
    return -G / r

A_z = build_Az(x_obs, x_i, z_i)  # (M, 1)
A_V = build_AV(x_obs, x_i, z_i)  # (M, 1)

# ─────────────────────────────────────────────────────────────
# 2.1) Raw gravity profile: g_z(x) = A_z @ m  (m < 0 -> negative anomaly)
# ─────────────────────────────────────────────────────────────
gz_ms2 = A_z @ m_i          # (M,)
gz_mgal = gz_ms2 * MS2_TO_MGAL

# ─────────────────────────────────────────────────────────────
# 2.2) Same source, switched to potential form: V(x) = A_V @ m
# ─────────────────────────────────────────────────────────────
V_si = A_V @ m_i            # (M,), m^2/s^2

# ─────────────────────────────────────────────────────────────
# 2.3) Gradient of (2.1): numerical d(g_z)/dx of the profile just computed
# ─────────────────────────────────────────────────────────────
dgz_dx_ms2_per_m = np.gradient(gz_ms2, x_obs)  # (M,), 1/s^2
dgz_dx_E = dgz_dx_ms2_per_m * SI_TO_EOTVOS      # Eotvos

print(f"Equivalent point mass (deficit): {m_i[0]:.3e} kg")
print(f"Peak g_z (most negative):        {gz_mgal.min():.4f} mGal")
print(f"Peak |dg_z/dx|:                  {np.max(np.abs(dgz_dx_E)):.2f} E")
print(f"Half-width x_1/2 (approx):       {0.766*z_i[0]:.1f} m")

# ─────────────────────────────────────────────────────────────
# Plotting (shared implementation in lib/util.py)
# ─────────────────────────────────────────────────────────────
plot_gravity_potential_gradient(
    x_obs, x_i, z_i, r_body,
    gz_mgal, V_si, dgz_dx_E,
    OUTDIR, x_min, x_max, dpi=300,
    body_outline=body_outline, body_label="void cavity",
)
