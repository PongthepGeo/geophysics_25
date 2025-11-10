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
salt = 4500                       # background velocity in m/s
box_velocity = 1500               # velocity in m/s for the box

# Model size
ny, nx = 500, 500                 # model size

# Box parameters
box_start_x, box_end_x = 200, 300
box_start_y, box_end_y = 300, 400

# Time parameters
time_steps = [85, 95, 105, 115]   # snapshots of wave propagation (ms)
dt = 0.004                        # Temporal sampling interval in seconds

# Source parameters
freq = 25                         # Frequency of the source in Hz

# Spatial parameters
dx = 4.0                          # Spatial sampling interval in meters

# Output parameters
save_path = "image_out/box_wave_propagation.png"

#-----------------------------------------------------------------------------------------#
# NOTE Setup
#-----------------------------------------------------------------------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# NOTE Create a source location
source_location = torch.tensor([[[0, nx // 2]]]).to(device)

# NOTE Create a velocity model with box anomaly
vp = salt * torch.ones(ny, nx)
vp[box_start_y:box_end_y, box_start_x:box_end_x] = box_velocity  # Box with lower velocity
vp = vp.to(device)

#-----------------------------------------------------------------------------------------#
# NOTE Plot the wave propagation with box overlay
#-----------------------------------------------------------------------------------------#

WAVE.box(vp, dx, dt, freq, time_steps, device, source_location,
         box_start_x, box_start_y, box_end_x, box_end_y, save_path)

#-----------------------------------------------------------------------------------------#