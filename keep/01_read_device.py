import pandas as pd
from lib.util import plot_acceleration_curves

# Step 1: Load the CSV
df = pd.read_csv("dataset/test.csv")

# Step 2: Select relevant columns
time = df["time"]
# ax = df["ax (m/s^2)"]
# ay = df["ay (m/s^2)"]
# az = df["az (m/s^2)"]
ax = df["Bx"]
ay = df["By"]
az = df["Bz"]

plot_acceleration_curves(time, ax, ay, az)