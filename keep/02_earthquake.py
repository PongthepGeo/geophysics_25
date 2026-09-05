import numpy as np
import matplotlib.pyplot as plt
from lib.util import loss, plot_epicenter
from scipy.optimize import minimize

stations = {'A': (0, 0), 'B': (2, 0), 'C': (0, 2)}
arrival_times = {'A': 1.628, 'B': 1.634, 'C': 1.640}
v = 3500
t0 = 1.620
dists = {sta: v * (arrival_times[sta] - t0) for sta in stations}
res = minimize(loss, x0=(1.0, 1.0), args=(stations, dists))
epicenter = res.x

plot_epicenter(stations, dists, epicenter)
