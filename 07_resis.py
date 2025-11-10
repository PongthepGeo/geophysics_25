#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERT forward modelling — dipole–dipole, 3 layers + polygon anomaly.
- Model display: (0,0) appears at TOP-LEFT (invert ONLY model axis).
- Pseudosection: NOT inverted.
- Headless; saves PNG only.

Files:
  forward_3layer.dat
  figure_out/ert_forward_3layer.png
"""
import os
import numpy as np

# Headless plotting (must be before importing pyplot)
import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

import pygimli as pg
from pygimli.physics import ert

#-----------------------------------------------------------------------------------------#
# NOTE Input Variables
#-----------------------------------------------------------------------------------------#

# Random seed
rng_seed = 1337

# Domain parameters (top-left is (0,0); y increases downward)
x_min, x_max, dx = 0.0, 120.0, 0.5     # wider than array to reduce edge effects
y_min, y_max, dy = 0.0, 60.0, 1.0      # depth to 60 m

# Layer parameters (depths are meters below surface y=0)
d1, d2 = 10.0, 30.0                    # Layer boundaries: 0–10, 10–30, 30–60 m
rho1, rho2, rho3 = 10.0, 50.0, 0.1  # Resistivity of each layer (Ω·m)
rho_anom = 1500                        # Resistivity of polygon anomaly (Ω·m)

# Polygon anomaly vertices (inside layer 2: 10–30 m depth)
poly_xy = np.array([
    [45.0, 12.0],
    [52.0, 14.0],
    [60.0, 13.0],
    [63.0, 20.0],
    [59.0, 26.0],
    [50.0, 24.0],
    [46.5, 19.0]
])

# Survey parameters
n_elecs = 41                           # Number of electrodes
elec_x_start = 10.0                    # First electrode position (m)
elec_x_end = 90.0                      # Last electrode position (m)
scheme_name = "dd"                     # Dipole-dipole configuration

# Simulation parameters
noise_level = 0.0                      # Noise level (relative)
noise_abs = 0.0                        # Absolute noise

# Output parameters
output_data_file = "forward_3layer.dat"
output_figure_dir = "figure_out"
output_figure_file = "figure_out/ert_forward_3layer.png"
figure_dpi = 300

#-----------------------------------------------------------------------------------------#

def main():
    # ----------------------------
    # Domain & grid setup
    # ----------------------------
    x_nodes = np.arange(x_min, x_max + dx*0.5, dx)
    y_nodes = np.arange(y_min, y_max + dy*0.5, dy)
    mesh = pg.createGrid(x=x_nodes, y=y_nodes)

    # ----------------------------
    # Create resistivity model
    # ----------------------------
    centers = np.array([[c.x(), c.y()] for c in mesh.cellCenters()])
    xc, yc = centers[:, 0], centers[:, 1]

    # Background layered model
    res = np.where(yc <= d1, rho1,
                   np.where(yc <= d2, rho2, rho3)).astype(float)

    # Add polygon anomaly inside layer 2
    path = Path(poly_xy, closed=True)
    inside_poly = path.contains_points(centers)
    in_second_layer = (yc > d1) & (yc <= d2)
    res[inside_poly & in_second_layer] = rho_anom

    # ----------------------------
    # Survey setup
    # ----------------------------
    elec_x = np.linspace(elec_x_start, elec_x_end, n_elecs)
    scheme = ert.createData(elecs=elec_x, schemeName=scheme_name)

    # ----------------------------
    # Forward simulation (no inversion)
    # ----------------------------
    data = ert.simulate(mesh=mesh, scheme=scheme, res=res,
                        noiseLevel=noise_level, noiseAbs=noise_abs, seed=rng_seed)

    # Clean negative rhoa if any (shouldn't happen with zero noise)
    data.remove(data["rhoa"] < 0)

    # Save data
    data.save(output_data_file)

    # ----------------------------
    # Plot → PNG
    #   - Top: true model (invert y so (0,0) shows at top-left)
    #   - Bottom: apparent-resistivity pseudosection (no inversion)
    # ----------------------------
    os.makedirs(output_figure_dir, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))

    # (a) True model
    ax1 = plt.subplot(2, 1, 1)
    gci, cbar = pg.show(mesh, data=res, ax=ax1, logScale=True, cMap="Spectral_r",
                        cMin=min(rho1, rho2, rho3, rho_anom),
                        cMax=max(rho1, rho2, rho3, rho_anom),
                        label="True resistivity (Ω·m)", orientation="vertical",
                        showMesh=False)
    ax1.set_title("True 3-layer model with polygon anomaly  (display top-left = (0,0))")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m, depth)")
    ax1.invert_yaxis()  # only for the model display

    # Layer boundaries
    for yb in (d1, d2):
        ax1.plot([x_nodes.min(), x_nodes.max()], [yb, yb], "k--", linewidth=0.9)
    # Polygon outline for reference
    ax1.add_patch(PathPatch(path, fill=False, edgecolor="k", linestyle="--", linewidth=1.2))

    # (b) Apparent-resistivity pseudosection (not inverted)
    ax2 = plt.subplot(2, 1, 2)
    # Set display range to percentiles to emphasize local lows from anomaly
    p5, p95 = np.percentile(data["rhoa"], [5, 95])
    ert.show(data, ax=ax2, logScale=True, cMap="Spectral_r",
             cMin=p5, cMax=p95, orientation="vertical")
    ax2.set_title("Apparent resistivity pseudosection (dipole–dipole)")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("Pseudo-depth")

    plt.tight_layout()
    plt.savefig(output_figure_file, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)

    # Quick console hint
    pg.info(f"Saved {output_figure_file}")


if __name__ == "__main__":
    main()
