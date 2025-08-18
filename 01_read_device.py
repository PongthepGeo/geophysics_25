import pandas as pd
from lib.util import plot_acceleration_curves

# Step 1: Load the CSV
df = pd.read_csv("dataset/seismic/location_b.csv")

# Step 2: Select relevant columns
time = df["time"]
ax = df["ax (m/s^2)"]
ay = df["ay (m/s^2)"]
az = df["az (m/s^2)"]

plot_acceleration_curves(time, ax, ay, az)