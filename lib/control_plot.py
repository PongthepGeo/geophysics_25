r"""Global matplotlib style shared by every figure script in 01_code.

The dpi/font-family core is mirrored from the manuscript figure code at
/home/laptop_pt/Desktop/work/paper/03_thin_section/manuscript/figure_code/control_plot.py
so class slides and paper figures share the same look. Two class-specific
additions sit on top of that shared core (not present in the paper file):

  - "figure.figsize": a 16:9 default (matches the beamer aspectratio=169
    slides) sized to 1152x648 px at 96 dpi -- big enough to read comfortably
    on a laptop/monitor, small enough to still fit alongside other windows.
    This is the answer to "can we globally define figsize instead of typing
    figsize=(10, 5) everywhere": any plt.figure()/plt.subplots() call made
    *without* an explicit figsize= picks this up automatically. Calls that
    still pass figsize= (multi-panel subplots, square image grids, wide
    image overlays, etc.) are intentionally keeping their own aspect ratio
    and should stay that way -- one global size can't fit every layout.
  - Larger label/tick/legend/title sizes than the paper file, since text
    at the paper's sizes reads too small once shrunk into a beamer slide.
  - "text.usetex": True -- labels/titles/legends are typeset by a real
    LaTeX install (matching the beamer slides), not matplotlib's built-in
    mathtext. This means any text handed to a title/label/legend call is
    LaTeX source, not a plain string: escape LaTeX-special characters
    that appear outside math mode ($, %, &, #, _, ^, {, }, ~, \\) and put
    real math in $...$ (e.g. r"$g_z$", not "g_z"). Plain non-ASCII text
    (e.g. literal "σ" or "Ω" typed directly, an en dash "–") will fail to
    compile -- use the LaTeX command instead (r"$\sigma$", r"$\Omega$",
    "--" for an en dash).

Usage:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1] / "lib"))

    import matplotlib.pyplot as plt
    from control_plot import PLOT_PARAMS

    plt.rcParams.update(PLOT_PARAMS)
"""

PLOT_PARAMS = {
    "savefig.dpi": 96,
    "figure.dpi": 96,
    "figure.figsize": (12, 6.75),
    "axes.labelsize": 20,
    "axes.labelweight": "bold",
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times"],
    "mathtext.fontset": "stix",
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{mathptmx}",  # Times-like text+math, matches font.serif above
}
