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
limestone = 3300                  # velocity in m/s

# Model size
ny, nx = 500, 500                 # model size

# Time parameters
time_steps = [50, 100, 120, 160]  # snapshot of wave propagation (ms)
dt = 0.004                        # Temporal sampling interval in seconds

# Source parameters
freq = 25                         # Frequency of the source in Hz

# Spatial parameters
dx = 4.0                          # Spatial sampling interval in meters

# Output parameters
save_path = "image_out/two_layers_wave_propagation.png"

#-----------------------------------------------------------------------------------------#
# NOTE Setup
#-----------------------------------------------------------------------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# NOTE Create a source location
source_location = torch.tensor([[[0, nx // 2]]]).to(device)

# NOTE Create a velocity model (two layers)
vp = sandstone * torch.ones(ny, nx)
vp[int(ny // 2):, :] = limestone  # Bottom half is limestone
vp = vp.to(device)

#-----------------------------------------------------------------------------------------#
# NOTE Plot the wave propagation
#-----------------------------------------------------------------------------------------#

WAVE.plot_wave_propagation(vp, dx, dt, freq, time_steps, device, source_location, save_path)

#-----------------------------------------------------------------------------------------#