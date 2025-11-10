#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./lib')
from WAVE import WAVE
#-----------------------------------------------------------------------------------------#
import numpy as np
from PIL import Image
import torch
import deepwave
#-----------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------#
# NOTE Input Variables
#-----------------------------------------------------------------------------------------#

# Image and velocity parameters
image_path = 'dataset/salt/sigsbee_salt.png'
minimum_velocity = 2000           # minimum velocity in m/s
maximum_velocity = 4700           # maximum velocity in m/s
# smooth = 20                       # Gaussian smoothing factor (higher = smoother, reduces scattering)
smooth = 5                       # Gaussian smoothing factor (higher = smoother, reduces scattering)

# Time parameters
nt = 400                          # Number of time steps
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

# Input/Output parameters
npy_folder = 'npy_folder'         # Load shot data from this folder

# Optimization parameters
optimizer_name = 'Adam'
lr = 1e-4
loss_fn_name = 'MSELoss'
n_epochs = 1

# Output parameters
clip_percent = 95                 # Percentile for clipping migration result
migration_npy_path = 'image_out/migration.npy'
migration_save_path = 'image_out/migrated_image.png'

#-----------------------------------------------------------------------------------------#
# NOTE Setup
#-----------------------------------------------------------------------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
dtype = torch.float

# NOTE Image to velocity model conversion
img = Image.open(image_path)
img_arr = np.array(img)
vp_array = WAVE.plot_velocity(img_arr, smooth, minimum_velocity, maximum_velocity)
nx = vp_array.shape[1]

# NOTE Create velocity model
vp = torch.tensor(vp_array, dtype=dtype).to(device)
vp = torch.transpose(vp, 0, 1)  # Transpose the model
vp = vp.to(device)

# NOTE Source parameters
n_shots = nx // shot_interval  # Number of shots based on model width
source_locations = torch.zeros(1, n_sources_per_shot, 2, dtype=dtype, device=device)
source_locations[..., 1] = source_depth

# NOTE Receiver locations
n_receivers_per_shot = nx // d_receiver
receiver_locations = torch.zeros(1, n_receivers_per_shot, 2, dtype=dtype, device=device)
receiver_locations[..., 1] = receiver_depth
receiver_locations[0, :, 0] = torch.arange(0, nx, d_receiver).long()[:n_receivers_per_shot]

# NOTE Source amplitudes
source_amplitudes = deepwave.wavelets.ricker(freq, nt, dt, peak_time).reshape(1, 1, -1).to(dtype=dtype, device=device)

#-----------------------------------------------------------------------------------------#
# NOTE Computing migration
#-----------------------------------------------------------------------------------------#

WAVE.run_migration(vp, npy_folder, dtype, device, dx, dt, source_amplitudes, receiver_locations, freq,
                   optimizer_name, lr, loss_fn_name, n_epochs, shot_interval, n_shots, migration_npy_path)

#-----------------------------------------------------------------------------------------#
# NOTE Plot migration result
#-----------------------------------------------------------------------------------------#

WAVE.plot_migration(migration_npy_path, clip_percent, migration_save_path)

#-----------------------------------------------------------------------------------------#