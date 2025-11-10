#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./lib')
from WAVE import WAVE
#-----------------------------------------------------------------------------------------#
import numpy as np
from PIL import Image
import torch
import os
from tqdm import tqdm
import shutil
#-----------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------#
# NOTE Input Variables
#-----------------------------------------------------------------------------------------#

# Image and velocity parameters
image_path = 'dataset/salt/sigsbee_salt.png'
minimum_velocity = 2000           # minimum velocity in m/s
maximum_velocity = 4700           # maximum velocity in m/s
smooth = 5                        # Gaussian smoothing factor (higher = smoother, reduces scattering)

# Time parameters
nt = 400                          # Number of time steps (how long wave propagates)
dt = 0.004                        # Temporal sampling interval in seconds

# Source parameters
freq = 25                         # Frequency of the source in Hz
peak_time = 1.5 / freq            # Time at which the Ricker wavelet reaches its peak
shot_interval = 10                # Every 10 pixels will allocate 1 shot
source_depth = 2                  # Source depth in grid points
n_sources_per_shot = 1

# Receiver parameters
d_receiver = 3                    # Receiver interval (grid points)
receiver_depth = 0

# Spatial parameters
dx = 4.0                          # Spatial sampling interval in meters

# Output parameters
output_dir = 'npy_folder'         # Directory to save receiver data

#-----------------------------------------------------------------------------------------#
# NOTE Setup
#-----------------------------------------------------------------------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
dtype = torch.float

# Clean and create output directory
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    print(f"Removed {output_dir}")
os.makedirs(output_dir)

# NOTE Image to velocity model conversion
img = Image.open(image_path)
img_arr = np.array(img)
vp_array = WAVE.plot_velocity(img_arr, smooth, minimum_velocity, maximum_velocity)
nx = vp_array.shape[1]

# NOTE Create velocity model
vp = torch.tensor(vp_array, dtype=dtype).to(device)
vp = torch.transpose(vp, 0, 1)  # Transpose the model
vp = vp.to(device)

# NOTE Source locations (one source per shot)
n_shots = nx // shot_interval  # Number of shots based on model width
source_locations = torch.zeros(1, n_sources_per_shot, 2, dtype=dtype, device=device)
source_locations[..., 1] = source_depth

# NOTE Receiver locations (distributed across model width)
n_receivers_per_shot = nx // d_receiver
receiver_locations = torch.zeros(1, n_receivers_per_shot, 2, dtype=dtype, device=device)
receiver_locations[..., 1] = receiver_depth
receiver_locations[0, :, 0] = torch.arange(0, nx, d_receiver).long()[:n_receivers_per_shot]

#-----------------------------------------------------------------------------------------#
# NOTE Compute wave propagation for each shot
#-----------------------------------------------------------------------------------------#

for i in tqdm(range(n_shots), desc="Computing shots"):
    current_source_position = i * shot_interval
    if current_source_position >= nx:
        break
    source_locations[0, 0, 0] = current_source_position
    _, receiver_amplitudes = WAVE.loop_wavefield(freq, nt, dt, peak_time, n_sources_per_shot, dtype, device,
                                                 vp, dx, source_locations, receiver_locations)
    np.save(os.path.join(output_dir, f'shot_pixel_{current_source_position:04d}.npy'),
            receiver_amplitudes.cpu().numpy())

#-----------------------------------------------------------------------------------------#