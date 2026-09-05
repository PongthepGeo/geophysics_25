# Enhanced demo: single profile figure (5 sources + total) and a geometry figure
# - Uses user's equation:
#     g_z(x0') = G * sum_i [ m_i * z_i / ((x0'-x_i)^2 + z_i^2)^(3/2) ]
# - Observation line: z' = 0 (surface), downward is positive (z_i > 0)
# - Outputs: two SVG figures, no CSV

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lib.util import plot_gravity

# Output folder (saved where you can download)
OUTDIR = Path("halfspace_gz_demo")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Constants
G = 6.67430e-11  # m^3 kg^-1 s^-2
MS2_TO_MGAL = 1e5  # 1 m/s^2 = 1e5 mGal

# Observation line (surface)
x_min, x_max, dx = -200.0, 200.0, 1.0
x_obs = np.arange(x_min, x_max + dx, dx)

# Five point sources (positions, depths, masses)
x_i = np.array([-5.0, 0.0, 7.0, 10.0, 150])  # m
z_i = np.array([60.0, 80.0, 60.0, 40.0, 10])     # m (downward +)
m_i = np.array([1.5e9, 3.0e9, 1.0e9, 2.5e9, 0.1e9])# kg

# Compute per-source and total gz (m/s^2)
gz_each = []
for xi, zi, mi in zip(x_i, z_i, m_i):
    dx_i = x_obs - xi
    r2 = dx_i**2 + zi**2
    r3 = r2 ** 1.5
    gz_i = G * mi * zi / r3  # m/s^2 (downward +)
    gz_each.append(gz_i)
gz_each = np.vstack(gz_each)
gz_total = gz_each.sum(axis=0)

# Convert to mGal
gz_each_mgal = gz_each * MS2_TO_MGAL
gz_total_mgal = gz_total * MS2_TO_MGAL

plot_gravity(gz_each_mgal, gz_total_mgal, x_obs, x_i, z_i, OUTDIR, x_min, x_max)