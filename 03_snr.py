import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from lib.util import SNR

# Step 1: Load the RGB image
img_path = "dataset/SNR/dike_fold.png"
img = Image.open(img_path).convert('RGB')
img_array = np.array(img)  # shape: (H, W, 3)

# Step 2: Choose vertical slice index (middle of image)
# x_index = img_array.shape[1] // 2  # vertical column
x_index = 158

# Step 3: Extract red channel along the vertical line
red_channel = img_array[:, :, 0]  # Red channel
vector_1d = red_channel[:, x_index]  # Vertical slice

SNR(img_array, x_index, vector_1d)