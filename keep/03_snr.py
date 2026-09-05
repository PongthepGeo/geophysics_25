import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from lib.util import SNR

# Load RGB image
img_path = "dataset/SNR/dike_fold.png"
img = Image.open(img_path).convert('RGB')
img_array = np.array(img)

# Vertical slice index
x_index = 158

# Extract red channel
red_channel = img_array[:, :, 0]
vector_clean = red_channel[:, x_index]

# Add Gaussian noise
noise_std = 20
noise = np.random.normal(loc=0, scale=noise_std, size=vector_clean.shape)
vector_noisy = np.clip(vector_clean + noise, 0, 255).astype(np.uint8)

# Plot all
SNR(img_array, x_index, vector_clean, vector_noisy)