import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
import os
from matplotlib.patches import Circle
from pathlib import Path

params = {
	'savefig.dpi': 300,  
	'figure.dpi' : 100,
	'axes.labelsize':12,  
	'axes.titlesize':12,
	'axes.titleweight': 'bold',
	'legend.fontsize': 10,
	'xtick.labelsize':10,
	'ytick.labelsize':10,
	'font.family': 'serif',
	'font.serif': 'Times New Roman'
}
matplotlib.rcParams.update(params)

def plot_acceleration_curves(time, ax, ay, az):
    plt.figure(figsize=(10, 5))
    plt.plot(time, ax, label='ax (m/s²)')
    plt.plot(time, ay, label='ay (m/s²)')
    plt.plot(time, az, label='az (m/s²)')

    plt.title("Acceleration Components Over Time")
    plt.xlabel("Time")
    plt.ylabel("Acceleration (m/s²)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_epicenter(stations, dists, epicenter):
    """
    Plot station locations, distance circles, and estimated epicenter.

    Parameters:
    -----------
    stations : dict
        Dictionary of station coordinates, e.g., {'A': (x1, y1), ...}
    dists : dict
        Dictionary of distances from origin time to each station, in meters.
    epicenter : tuple
        Estimated epicenter coordinates (x, y)
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {'A': 'r', 'B': 'g', 'C': 'b'}

    for sta in stations:
        x, y = stations[sta]
        r = dists[sta]
        circle = Circle((x, y), r, color=colors.get(sta, 'gray'), alpha=0.3, label=f"{sta} (r = {r:.1f} m)")
        ax.add_patch(circle)
        ax.plot(x, y, 'o', color=colors.get(sta, 'gray'))
        ax.text(x + 0.1, y + 0.1, sta)

    # Plot epicenter
    x_epi, y_epi = epicenter
    ax.plot(x_epi, y_epi, 'k*', markersize=15, label='Epicenter')

    # Dynamic plot limits
    all_x = [p[0] for p in stations.values()] + [x_epi]
    all_y = [p[1] for p in stations.values()] + [y_epi]
    margin = max(dists.values()) * 0.2
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.set_aspect('equal')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Estimated Earthquake Epicenter from Arrival Times")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def loss(xy, stations, dists):
    x, y = xy
    return sum((np.sqrt((x - sx)**2 + (y - sy)**2) - dists[sta])**2
               for sta, (sx, sy) in stations.items())

def SNR(img_array, x_index, vector_clean, vector_noisy):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: RGB image with vertical line
    ax1.imshow(img_array)
    ax1.axvline(x=x_index, color='red', linestyle='--', label=f'x = {x_index}')
    ax1.set_title("RGB Image with Vertical Slice")
    ax1.axis('off')
    ax1.legend()

    # Panel 2: Clean red-channel profile
    y = np.arange(len(vector_clean))
    ax2.plot(vector_clean, y, color='red')
    ax2.set_title(f"Clean Red Channel Profile (x = {x_index})")
    ax2.set_xlabel("Intensity")
    ax2.set_ylabel("Vertical Pixel Position")
    ax2.invert_yaxis()

    # Panel 3: Noisy profile
    ax3.plot(vector_noisy, y, color='blue')
    ax3.set_title("With Gaussian Noise")
    ax3.set_xlabel("Intensity")
    ax3.set_ylabel("Vertical Pixel Position")
    ax3.invert_yaxis()

    plt.tight_layout()
    plt.show()

def plot_gravity(gz_each_mgal, gz_total_mgal, x_obs, x_i, z_i, OUTDIR, x_min, x_max):
    # 1) Single figure: all five sources + total on one plot
    plt.figure(figsize=(10, 5))
    for sidx in range(gz_each_mgal.shape[0]):
        plt.plot(x_obs, gz_each_mgal[sidx], linewidth=1.8, label=f"source {sidx+1}")
    # Plot total last for emphasis
    plt.plot(x_obs, gz_total_mgal, linewidth=2.6, linestyle="-", label="total")
    # Mark source x-positions
    for xi in x_i:
        plt.axvline(xi, linestyle="--", linewidth=1)
    plt.title("Vertical gravity $g_z$ along surface (5 sources + total)")
    plt.xlabel("x (m) at surface")
    plt.ylabel("g_z (mGal, downward +)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncols=3, frameon=True)
    fig_profile = OUTDIR / "gz_profile_all.png"
    plt.tight_layout()
    plt.savefig(fig_profile, format="png")
    plt.show()

    # 2) Geometry figure: half-space with 5 sources and gravimeters
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
    plt.legend(frameon=True)
    plt.tight_layout()
    fig_geom = OUTDIR / "geometry_sources_gravimeters.png"
    plt.savefig(fig_geom, format="png")
    plt.show()

    print("Saved figures:")
    print(" -", fig_profile)
    print(" -", fig_geom)