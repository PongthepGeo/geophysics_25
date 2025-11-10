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
sandstone = 2500                 # velocity in m/s

# Model size
ny, nx = 1000, 1000              # model size

# Time parameters
time_steps = [80, 120, 140, 180] # snapshot of wave propagation (ms)
dt = 0.004                       # Temporal sampling interval (time step) in seconds

# Source parameters
freq = 25                        # Frequency of the source in Hz

# Spatial parameters
dx = 4.0                         # Spatial sampling interval (distance between grid points) in meters

# Output parameters
save_path = "image_out/wave_propagation.png"

#-----------------------------------------------------------------------------------------#
# NOTE Setup
#-----------------------------------------------------------------------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# NOTE Create a source location
source_location = torch.tensor([[[ny // 2, nx // 2]]]).to(device)

# NOTE Create a velocity model
vp = sandstone * torch.ones(ny, nx)
vp = torch.transpose(vp, 0, 1)  # Transpose the model
vp = vp.to(device)

#-----------------------------------------------------------------------------------------#
# NOTE Plot the wave propagation
#-----------------------------------------------------------------------------------------#

WAVE.plot_wave_propagation(vp, dx, dt, freq, time_steps, device, source_location, save_path)

#-----------------------------------------------------------------------------------------#