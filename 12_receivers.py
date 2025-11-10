#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./lib')
from WAVE import WAVE
#-----------------------------------------------------------------------------------------#
import torch
#-----------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------#
# NOTE Input Variables
#-----------------------------------------------------------------------------------------#

# Velocity parameters
sandstone = 2700                  # velocity in m/s
salt = 3300                       # velocity in m/s

# Model size
ny, nx = 1000, 1000               # model size

# Layer parameters
limestone_start = 500             # depth for reference line

# Time parameters
time_steps = [250, 300]           # snapshot of wave propagation (ms)
dt = 0.004                        # Temporal sampling interval in seconds

# Source parameters
freq = 25                         # Frequency of the source in Hz
peak_time = 1.5 / freq            # The time at which the Ricker wavelet reaches its peak
n_shots = 1
n_sources_per_shot = 1
d_source = 1
first_source = int(nx // 2)
source_depth = 2

# Receiver parameters
d_receiver = 3                    # number of grid points between receivers (approximating 10-12 meters)
receiver_depth = 0

# Spatial parameters
dx = 4.0                          # Spatial sampling interval in meters

# Output parameters
save_path = "image_out/receivers_wave_propagation.png"

#-----------------------------------------------------------------------------------------#
# NOTE Setup
#-----------------------------------------------------------------------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# NOTE Create a velocity model (three layers)
salt_end = ny // 3
sandstone_end = 2 * ny // 3
vp = sandstone * torch.ones(ny, nx)
vp[:salt_end, :] = salt           # Top layer is salt
vp[salt_end:sandstone_end, :] = sandstone  # Middle layer is sandstone
vp = torch.transpose(vp, 0, 1)    # Transpose the model
vp = vp.to(device)

# NOTE Source locations (one source at the top center of the model)
source_locations = torch.zeros(n_shots, n_sources_per_shot, 2, dtype=torch.long, device=device)
source_locations[..., 1] = source_depth
source_locations[:, 0, 0] = (torch.arange(n_shots) * d_source + first_source)

# NOTE Receiver locations (approximately 10-12 meters receiver interval)
n_receivers_per_shot = nx // d_receiver
receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2, dtype=torch.long, device=device)
receiver_locations[..., 1] = receiver_depth
receiver_locations[:, :, 0] = torch.arange(0, nx, d_receiver).long()[:n_receivers_per_shot]

#-----------------------------------------------------------------------------------------#
# NOTE Compute and plot the wave propagation plus the receivers
#-----------------------------------------------------------------------------------------#

WAVE.plot_receivers(freq, dt, peak_time, n_shots, n_sources_per_shot, device, vp, dx,
					source_locations, receiver_locations, time_steps, limestone_start, save_path)

#-----------------------------------------------------------------------------------------#

