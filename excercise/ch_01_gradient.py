import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

PLOT_PARAMS = {
    "savefig.dpi": 300,
    "figure.dpi": 100,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
}

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
# Plot
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, g, linewidth=2.0, label=r"$g$",)
ax.plot(x, dg_dx, linewidth=2.0, label=r"$\partial g/\partial x$",)
ax.plot(x, dg_dy, linewidth=2.0, label=r"$\partial g/\partial y$",)
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