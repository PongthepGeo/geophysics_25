import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
import os
from matplotlib.patches import Circle
from pathlib import Path

import matplotlib
try:  # shared global plot style (lib/control_plot.py)
	from .control_plot import PLOT_PARAMS
except ImportError:
	from control_plot import PLOT_PARAMS
matplotlib.rcParams.update(PLOT_PARAMS)

# def plot_acceleration_curves(time, ax, ay, az):
#     plt.figure(figsize=(10, 5))
#     plt.plot(time, ax, label='ax (m/s²)')
#     plt.plot(time, ay, label='ay (m/s²)')
#     plt.plot(time, az, label='az (m/s²)')

#     plt.title("Acceleration Components Over Time")
#     plt.xlabel("Time")
#     plt.ylabel("Acceleration (m/s²)")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# def plot_epicenter(stations, dists, epicenter):
#     """
#     Plot station locations, distance circles, and estimated epicenter.

#     Parameters:
#     -----------
#     stations : dict
#         Dictionary of station coordinates, e.g., {'A': (x1, y1), ...}
#     dists : dict
#         Dictionary of distances from origin time to each station, in meters.
#     epicenter : tuple
#         Estimated epicenter coordinates (x, y)
#     """
#     fig, ax = plt.subplots(figsize=(6, 6))
#     colors = {'A': 'r', 'B': 'g', 'C': 'b'}

#     for sta in stations:
#         x, y = stations[sta]
#         r = dists[sta]
#         circle = Circle((x, y), r, color=colors.get(sta, 'gray'), alpha=0.3, label=f"{sta} (r = {r:.1f} m)")
#         ax.add_patch(circle)
#         ax.plot(x, y, 'o', color=colors.get(sta, 'gray'))
#         ax.text(x + 0.1, y + 0.1, sta)

#     # Plot epicenter
#     x_epi, y_epi = epicenter
#     ax.plot(x_epi, y_epi, 'k*', markersize=15, label='Epicenter')

#     # Dynamic plot limits
#     all_x = [p[0] for p in stations.values()] + [x_epi]
#     all_y = [p[1] for p in stations.values()] + [y_epi]
#     margin = max(dists.values()) * 0.2
#     ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
#     ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

#     ax.set_aspect('equal')
#     ax.set_xlabel("X (m)")
#     ax.set_ylabel("Y (m)")
#     ax.set_title("Estimated Earthquake Epicenter from Arrival Times")
#     ax.legend()
#     ax.grid(True)
#     plt.tight_layout()
#     plt.show()

# def loss(xy, stations, dists):
#     x, y = xy
#     return sum((np.sqrt((x - sx)**2 + (y - sy)**2) - dists[sta])**2
#                for sta, (sx, sy) in stations.items())

# def SNR(img_array, x_index, vector_clean, vector_noisy):
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

#     # Panel 1: RGB image with vertical line
#     ax1.imshow(img_array)
#     ax1.axvline(x=x_index, color='red', linestyle='--', label=f'x = {x_index}')
#     ax1.set_title("RGB Image with Vertical Slice")
#     ax1.axis('off')
#     ax1.legend()

#     # Panel 2: Clean red-channel profile
#     y = np.arange(len(vector_clean))
#     ax2.plot(vector_clean, y, color='red')
#     ax2.set_title(f"Clean Red Channel Profile (x = {x_index})")
#     ax2.set_xlabel("Intensity")
#     ax2.set_ylabel("Vertical Pixel Position")
#     ax2.invert_yaxis()

#     # Panel 3: Noisy profile
#     ax3.plot(vector_noisy, y, color='blue')
#     ax3.set_title("With Gaussian Noise")
#     ax3.set_xlabel("Intensity")
#     ax3.set_ylabel("Vertical Pixel Position")
#     ax3.invert_yaxis()

#     plt.tight_layout()
#     plt.show()

def plot_gravity(gz_each_mgal, gz_total_mgal, x_obs, x_i, z_i, OUTDIR, x_min, x_max):
    # 1) Single figure: all five sources + total on one plot (global figure.figsize default)
    plt.figure()
    for sidx in range(gz_each_mgal.shape[0]):
        plt.plot(x_obs, gz_each_mgal[sidx], linewidth=1.8, label=f"source {sidx+1}")
    # Plot total last for emphasis
    plt.plot(x_obs, gz_total_mgal, linewidth=2.6, linestyle="-", label="total")
    # Mark source x-positions
    for xi in x_i:
        plt.axvline(xi, linestyle="--", linewidth=1)
    plt.title("Vertical gravity $g_z$ along surface (5 sources + total)")
    plt.xlabel("x (m) at surface")
    plt.ylabel(r"$g_z$ (mGal, downward +)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncols=2, frameon=True, loc="upper right")
    fig_profile = OUTDIR / "gz_profile_all.png"
    plt.tight_layout()
    plt.savefig(fig_profile, format="png")
    plt.show()

    # 2) Geometry figure: half-space with 5 sources and gravimeters
    # (short/wide cross-section -- intentionally not the global 16:9 default)
    plt.figure(figsize=(10, 3.6))
    # Surface line (z=0)
    plt.hlines(0.0, x_min, x_max, linestyles="-", linewidth=2)
    # Gravimeter stations (subsample to reduce clutter)
    gstep = 25  # meters (show station every 25 m)
    gpos = x_obs[::gstep]
    plt.scatter(gpos, np.zeros_like(gpos), marker="^", s=20, label="gravimeters")
    # Point sources
    plt.scatter(x_i, z_i, marker="o", s=50, label="point sources")
    for idx, (xi, zi) in enumerate(zip(x_i, z_i), start=1):
        plt.text(xi, zi, f"  m{idx}", va="center")
    # Axes and labels
    plt.xlim(x_min, x_max)
    zmax = max(z_i.max() + 20.0, 60.0)
    plt.ylim(0.0, zmax)
    ax = plt.gca()
    ax.invert_yaxis()  # show depth increasing downward
    plt.xlabel("x (m)")
    plt.ylabel("z (m, downward +)")
    plt.title("Half-space geometry: 5 sources and surface gravimeters")
    plt.legend(ncols=2, frameon=True, loc="upper right")
    plt.tight_layout()
    fig_geom = OUTDIR / "geometry_sources_gravimeters.png"
    plt.savefig(fig_geom, format="png")
    plt.show()

    print("Saved figures:")
    print(" -", fig_profile)
    print(" -", fig_geom)

def plot_gravity_potential_gradient(x_obs: np.ndarray,
                                     x_i: np.ndarray,
                                     z_i: np.ndarray,
                                     r_body: float,
                                     gz_mgal: np.ndarray,
                                     V_si: np.ndarray,
                                     dgz_dx_E: np.ndarray,
                                     OUTDIR: Path,
                                     x_min: float,
                                     x_max: float,
                                     dpi: int = 300,
                                     body_outline: np.ndarray = None,
                                     body_label: str = "buried body") -> None:
    """Buried circular body: geometry + the V -> g_z -> dg_z/dx chain
    (used by ch_02_04_gravity_matrix_potential_gradient.py).

    All three profiles come from the *same* buried body, so plotting them
    stacked on a shared x-axis shows directly how each derivative narrows
    the anomaly -- the same point made in the "Chain of Resolution" slide.

    body_outline: optional (K,2) array of (x,z) points describing an
    irregular, "circle-like" outline to draw instead of a perfect Circle
    (e.g. a natural dissolution cavity) -- the physics still uses the
    simple equivalent-point-mass approximation regardless of which shape
    is drawn; only the geometry panel's artwork changes.
    """
    OUTDIR = Path(OUTDIR)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # 1) Geometry: half-space with the buried body + surface stations
    # (short/wide cross-section -- intentionally not the global 16:9 default)
    plt.figure(figsize=(10, 3.6))
    plt.hlines(0.0, x_min, x_max, linestyles="-", linewidth=2, label="surface (z=0)")
    gstep = 25
    gpos = x_obs[::gstep]
    plt.scatter(gpos, np.zeros_like(gpos), marker="^", s=20, label="gravimeters")
    ax = plt.gca()
    if body_outline is not None:
        poly = plt.Polygon(body_outline, closed=True, facecolor="0.6",
                            edgecolor="black", alpha=0.7, label=body_label)
        ax.add_patch(poly)
    else:
        for xi, zi in zip(x_i, z_i):
            circ = Circle((xi, zi), r_body, facecolor="0.6", edgecolor="black",
                           alpha=0.7, label=body_label)
            ax.add_patch(circ)
    plt.xlim(x_min, x_max)
    zmax = float(z_i.max() + r_body + 60.0)
    plt.ylim(0.0, zmax)
    ax.invert_yaxis()  # depth increases downward
    plt.xlabel("x (m)")
    plt.ylabel("z (m, downward +)")
    plt.title("Buried body and surface gravimeters (shape not to scale in x)")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig_geom = OUTDIR / "geometry_body.png"
    plt.savefig(fig_geom, format="png", dpi=dpi)
    plt.show()
    plt.close()

    # 2) Three stacked profiles: potential -> field -> gradient, same x-axis
    # (tall multi-panel -- intentionally not the global 16:9 default)
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, figsize=(10, 10))

    ax0.plot(x_obs, V_si, color="tab:blue", linewidth=2.2)
    ax0.axvline(0.0, linestyle="--", color="gray", linewidth=1)
    ax0.set_ylabel(r"$V$ (m$^2$/s$^2$)")
    ax0.set_title(r"Potential $V(x) = -Gm/r$ --- broad and smooth")
    ax0.grid(True, alpha=0.3)

    ax1.plot(x_obs, gz_mgal, color="tab:orange", linewidth=2.2)
    ax1.axvline(0.0, linestyle="--", color="gray", linewidth=1)
    ax1.set_ylabel(r"$g_z$ (mGal)")
    ax1.set_title(r"Field $g_z(x) = Gmz/r^3$ --- narrower, peaks directly over the body")
    ax1.grid(True, alpha=0.3)

    ax2.plot(x_obs, dgz_dx_E, color="tab:purple", linewidth=2.2)
    ax2.axvline(0.0, linestyle="--", color="gray", linewidth=1)
    ax2.set_ylabel("$\\partial g_z/\\partial x$ (E)")
    ax2.set_xlabel("x (m)")
    ax2.set_title(r"Gradient $\partial g_z/\partial x$ --- sharp: pinpoints the body's edges")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_profiles = OUTDIR / "potential_field_gradient.png"
    plt.savefig(fig_profiles, format="png", dpi=dpi)
    plt.show()
    plt.close(fig)

    print("Saved figures:")
    print(" -", fig_geom)
    print(" -", fig_profiles)

def plot_gravity_matrix(gz_each_mgal: np.ndarray,
                         gz_total_mgal: np.ndarray,
                         x_obs: np.ndarray,
                         x_i: np.ndarray,
                         z_i: np.ndarray,
                         OUTDIR: Path,
                         x_min: float,
                         x_max: float,
                         dpi: int = 300) -> None:
    """Matrix-form gravity plot (used by ch_02_02_gravity_matrix.py).

    gz_each_mgal has shape (M, N): rows are surface stations, columns are
    source cells -- unlike plot_gravity() above, whose gz_each_mgal is (N, M).
    """
    OUTDIR = Path(OUTDIR)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # 1) Profile: all sources + total (uses the global figure.figsize default)
    plt.figure()
    N = gz_each_mgal.shape[1]
    for sidx in range(N):
        plt.plot(x_obs, gz_each_mgal[:, sidx], linewidth=1.8, label=f"source {sidx+1}")
    plt.plot(x_obs, gz_total_mgal, linewidth=2.6, linestyle="-", label="total")

    for xi in x_i:
        plt.axvline(xi, linestyle="--", linewidth=1)

    plt.title(r"Vertical gravity $g_z$ along surface (matrix form: $g_z=A_z\,\sigma$)")
    plt.xlabel("x (m) at surface")
    plt.ylabel(r"$g_z$ (mGal, downward +)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncols=2, frameon=True, loc="upper left")
    fig_profile = OUTDIR / "gz_profile_all.png"
    plt.tight_layout()
    plt.savefig(fig_profile, format="png", dpi=dpi)
    plt.show()

    # 2) Geometry figure (short/wide cross-section -- intentionally not
    # the global 16:9 default, which would waste vertical space here)
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
    plt.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    fig_geom = OUTDIR / "geometry_sources_gravimeters.png"
    plt.savefig(fig_geom, format="png", dpi=dpi)
    plt.show()

    print("Saved figures:")
    print(" -", fig_profile)
    print(" -", fig_geom)