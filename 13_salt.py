#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./lib')
from WAVE import WAVE
#-----------------------------------------------------------------------------------------#
from PIL import Image
import numpy as np
import torch
#-----------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------#
# NOTE Input Variables
#-----------------------------------------------------------------------------------------#

# Image and velocity parameters
image_path = 'dataset/salt/sigsbee_salt.png'
minimum_velocity = 2000           # minimum velocity in m/s
maximum_velocity = 4700           # maximum velocity in m/s
resize_factor = 2                 # resize image by dividing by this factor

# Time parameters
time_steps = [50, 75, 100, 120]  # snapshot of wave propagation (ms)
dt = 0.004                        # Temporal sampling interval in seconds

# Source parameters
freq = 25                         # Frequency of the source in Hz

# Spatial parameters
dx = 4.0                          # Spatial sampling interval in meters

# Output parameters
velocity_save_path = 'image_out/salt_velocity_model.png'
wave_save_path = 'image_out/salt_wave_propagation.png'

#-----------------------------------------------------------------------------------------#
# NOTE Setup
#-----------------------------------------------------------------------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
dtype = torch.float

# NOTE Import the image and convert it to velocity model
img = Image.open(image_path)
width, height = img.size
img_resized = img.resize((width // resize_factor, height // resize_factor))
img_arr = np.array(img_resized)

# Convert image to velocity model
velocity_array = WAVE.photo2velocity(img_arr, minimum_velocity, maximum_velocity, velocity_save_path)

# NOTE Create velocity model and locate source
ny, nx = velocity_array.shape
source_location = torch.tensor([[[0, nx // 2]]]).to(device)
vp = torch.tensor(velocity_array, dtype=dtype).to(device)

#-----------------------------------------------------------------------------------------#
# NOTE Compute wave propagation and plot snapshots of wave propagation
#-----------------------------------------------------------------------------------------#

WAVE.plot_wave_propagation_dtype(vp, dx, dt, freq, time_steps, dtype, device, source_location, wave_save_path)

#-----------------------------------------------------------------------------------------#