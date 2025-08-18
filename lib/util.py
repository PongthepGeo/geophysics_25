import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
import os
from matplotlib.patches import Circle

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