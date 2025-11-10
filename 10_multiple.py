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
sandstone = 4500                  # velocity in m/s
salt = 2500                       # velocity in m/s

# Model size
ny, nx = 500, 500                 # model size

# Time parameters
time_steps = [50, 70, 140, 200]   # snapshot of wave propagation (ms)
dt = 0.004                        # Temporal sampling interval in seconds

# Source parameters
freq = 25                         # Frequency of the source in Hz

# Spatial parameters
dx = 4.0                          # Spatial sampling interval in meters

# Output parameters
save_path = "image_out/multiple_layers_wave_propagation.png"

#-----------------------------------------------------------------------------------------#
# NOTE Setup
#-----------------------------------------------------------------------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# NOTE Create a source location
source_location = torch.tensor([[[0, nx // 2]]]).to(device)

# NOTE Create a velocity model (three layers: salt-sandstone-salt)
salt_end = ny // 3
sandstone_end = 2 * ny // 3
vp = sandstone * torch.ones(ny, nx)
vp[:salt_end, :] = salt           # Top layer is salt
vp[salt_end:sandstone_end, :] = sandstone  # Middle layer is sandstone
vp[sandstone_end:, :] = salt      # Bottom layer is salt
vp = vp.to(device)

#-----------------------------------------------------------------------------------------#
# NOTE Plot the wave propagation
#-----------------------------------------------------------------------------------------#

WAVE.plot_wave_propagation(vp, dx, dt, freq, time_steps, device, source_location, save_path)

#-----------------------------------------------------------------------------------------#
