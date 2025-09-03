#!/usr/bin/env python3
"""
2D fixed-frame gravity (downward +) from a buried circular anomaly.

Frame:
  - Discrete 2D surface grid of size (NY, NX) = (200, 200)
  - Pixel coordinates: top-left (0,0), x to the right, y downward (image-style)
  - z is vertical, positive downward

Mass model:
  - A circular patch ("anomaly") at constant depth z = depth_m (>0, downward)
  - Density contrast rho_contrast (kg/m^3) * thickness_m (m) -> surface density σ (kg/m^2)
  - Outside the circle: σ = 0

Forward model:
  - Each mass pixel is a point mass dm = σ * dA at (x', y', z=depth_m)
  - Observer pixels are at the surface z=0
  - Vertical gravity (downward +):
        g_z = G * dm * depth_m / ( (dx^2 + dy^2 + depth_m^2)^(3/2) )
  - Sum contributions from all mass pixels for each observer pixel

Outputs:
  - figure_out/gz_map.svg : 2D map of Δg_z (mGal), origin='upper'
  - figure_out/gz_profile.svg : center-row profile (mGal)
  - npy dumps for reuse
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

# Gravitational constant (SI)
G = 6.67430e-11  # m^3 kg^-1 s^-2

# ----------------------------
# 1) Frame & anomaly settings
# ----------------------------
NX, NY = 200, 200                 # fixed frame 200x200 pixels
px_size_m = 10.0                  # meters per pixel (set your scale)
depth_m = 400.0                   # depth (downward positive)
rho_contrast = 400.0              # kg/m^3 (inside circle)
thickness_m = 20.0                # m -> σ = ρ * t

# Circle in pixel coordinates (cx, cy) with radius in pixels
cx_px, cy_px = 100, 100           # center (put it anywhere inside the frame)
radius_px = 30                    # in pixels

# ----------------------------
# 2) Build coordinates (top-left origin, y down)
# ----------------------------
# Pixel-center coordinates in meters
x_coords = (np.arange(NX) + 0.5) * px_size_m
y_coords = (np.arange(NY) + 0.5) * px_size_m

# Meshgrid of observer pixel centers (surface, z=0)
XX, YY = np.meshgrid(x_coords, y_coords, indexing="xy")

# ----------------------------
# 3) Mass mask: circle with σ inside, 0 outside
# ----------------------------
j = np.arange(NX)  # x-index (cols)
i = np.arange(NY)  # y-index (rows)
JJ, II = np.meshgrid(j, i, indexing="xy")

circle_mask = (JJ - cx_px)**2 + (II - cy_px)**2 <= radius_px**2
sigma = np.zeros((NY, NX), dtype=float)
sigma[circle_mask] = rho_contrast * thickness_m  # kg/m^2 inside the circle

# ----------------------------
# 4) Prepare source list (mass pixels)
# ----------------------------
dA = px_size_m * px_size_m               # area per pixel (m^2)
src_y_idx, src_x_idx = np.where(circle_mask)
src_x = (src_x_idx + 0.5) * px_size_m    # meters
src_y = (src_y_idx + 0.5) * px_size_m    # meters
src_sigma = sigma[src_y_idx, src_x_idx]  # kg/m^2 (all identical here)
dm = src_sigma * dA                      # kg per source pixel
Ns = src_x.size

print(f"[info] Frame: {NX} x {NY} px, pixel={px_size_m} m, depth={depth_m} m")
print(f"[info] Circle center (px): ({cx_px},{cy_px}), radius={radius_px} px")
print(f"[info] Mass pixels: {Ns} (~πR^2), σ={rho_contrast*thickness_m:.2f} kg/m^2, dm (single)={dm[0]:.2e} kg")

# ----------------------------
# 5) Forward model: Δg_z at all surface pixels (downward +)
#     Chunk over observer pixels for memory safety
# ----------------------------
gz = np.zeros((NY, NX), dtype=float)  # m/s^2

# Flatten observers for chunk processing
obs_x_flat = XX.ravel()
obs_y_flat = YY.ravel()
K = obs_x_flat.size

chunk = 4000  # adjust if you have more/less memory
depth2 = depth_m * depth_m

for start in range(0, K, chunk):
    end = min(start + chunk, K)
    Ox = obs_x_flat[start:end][:, None]  # (k,1)
    Oy = obs_y_flat[start:end][:, None]  # (k,1)

    # Differences to all sources (broadcast)
    dx = Ox - src_x[None, :]            # (k, Ns)
    dy = Oy - src_y[None, :]            # (k, Ns)

    r2 = dx*dx + dy*dy + depth2         # (k, Ns)
    r32 = r2 * np.sqrt(r2)              # (k, Ns) => r^3

    # g_z contribution from all sources (downward positive):
    #   g_z = G * dm * depth / r^3
    # Sum over sources axis
    gz_block = G * np.sum(dm[None, :] * depth_m / r32, axis=1)  # (k,)

    gz.ravel()[start:end] = gz_block

# Convert to mGal (1 mGal = 1e-5 m/s^2)
gz_mgal = gz / 1e-5

# ----------------------------
# 6) Save figures & arrays
# ----------------------------
os.makedirs("figure_out", exist_ok=True)
np.save("figure_out/gz_map_mgal.npy", gz_mgal)
np.save("figure_out/sigma.npy", sigma)

# Map view
fig1, ax1 = plt.subplots(figsize=(6.8, 5.6))
im = ax1.imshow(gz_mgal, origin='upper', extent=[x_coords[0], x_coords[-1], y_coords[-1], y_coords[0]])
cb = fig1.colorbar(im, ax=ax1)
cb.set_label(r"$\Delta g_z$ (mGal, downward +)")
ax1.set_title("Gravity anomaly map from buried circular patch")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
# overlay the circle outline (in meters)
circ = Circle(((cx_px+0.5)*px_size_m, (cy_px+0.5)*px_size_m), radius_px*px_size_m,
              fill=False, linewidth=1.0)
ax1.add_patch(circ)
plt.tight_layout()
fig1.savefig("figure_out/gz_map.svg", format="svg", bbox_inches="tight",
             transparent=True, pad_inches=0.0)

# Center-row profile (y = center)
row = cy_px
profile = gz_mgal[row, :]
fig2, ax2 = plt.subplots(figsize=(7.2, 3.2))
ax2.plot(x_coords, profile, lw=2)
ax2.set_xlabel("x (m)")
ax2.set_ylabel(r"$\Delta g_z$ (mGal, downward +)")
ax2.set_title(f"Center-row profile (y={row}, depth={depth_m:.0f} m)")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
fig2.savefig("figure_out/gz_profile.svg", format="svg", bbox_inches="tight",
             transparent=True, pad_inches=0.0)

print("[done] Saved: figure_out/gz_map.svg, figure_out/gz_profile.svg")
