import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "lib"))
from control_plot import PLOT_PARAMS  # shared global plot style

matplotlib.rcParams.update(PLOT_PARAMS)

import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Synthetic gravity response
# ---------------------------------------------------------

x = np.linspace(-6, 6, 1000)
z = 2.0; y0 = 1.0

g = z / (x**2 + y0**2 + z**2)**1.5
dg_dx = (
    -3.0 * z * x
    / (x**2 + y0**2 + z**2)**2.5
)

dg_dy = (
    10 * -3.0 * z * y0
    / (x**2 + y0**2 + z**2)**2.5
)

# ---------------------------------------------------------
# Normalize for shape comparison
# ---------------------------------------------------------

g /= np.max(np.abs(g))
dg_dx /= np.max(np.abs(dg_dx))
dg_dy /= np.max(np.abs(dg_dy))

# ---------------------------------------------------------
# Enhance dg_dx
# ---------------------------------------------------------

# print(dg_dx)
# dg_dx_new = 2 * dg_dx
# preallocate
amplify = np.zeros_like(dg_dx)
# print(amplify)
# amplify = dg_dx
# amplify[500:] = dg_dx[:500] + dg_dx[500:] * 2 
amplify[500:] = dg_dx[500:] * 2 
amplify[:500] = dg_dx[:500] 

max_amp = np.max(np.abs(amplify[:500]))
print(max_amp)

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

fig, ax = plt.subplots()  # uses the global figure.figsize default
ax.plot(x, g, linewidth=2.0, label=r"$g$",)
ax.plot(x, dg_dx, linewidth=2.0, label=r"$\partial g/\partial x$",)
ax.plot(x, dg_dy, linewidth=2.0, label=r"$\partial g/\partial y$",)
ax.plot(x, amplify, linewidth=2.0, linestyle="--", label=r"amplify $\partial g/\partial y$",)
ax.axhline(0, linewidth=0.8, linestyle="--",)
ax.set_title("Synthetic Gravity Response and Its Horizontal/Vertical Gradients")
ax.set_xlabel(r"$x$ (km)")
ax.set_ylabel("Normalized amplitude")
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("gravity_gradient_overlay.png", bbox_inches="tight",)
plt.show()