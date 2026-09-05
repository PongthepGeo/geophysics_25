import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# =============================
# PLOT STYLE
# =============================
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "lib"))
from control_plot import PLOT_PARAMS  # shared global plot style

matplotlib.rcParams.update(PLOT_PARAMS)


# =============================
# CONFIGURATION
# =============================
MIGRATION_NPY = "/home/laptop_pt/Desktop/geophysics_25/01_code/image_out/migration.npy"
TOP_CLIP_COLS = 52

X_TRACE = 1250
Y_START = 780

PREWHITEN = 2e-3
WAVELET_HALFWIN = 45
WAVELET_X_HALFWIDTH = 40

TRACE_HP_WIN = 151
REFLECT_DETREND_WIN = 31
STEM_KEEP_FRAC = 0.10

AI_TARGET_RMAX = 0.10
AI_Q = 99.0
Z0 = 9000.0

# AI overlay settings
AI_STRIP_WIDTH = 30  # width of AI strip in pixels for overlay


# =============================
# Basic helpers
# =============================
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def zscore(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - float(np.mean(x))) / (float(np.std(x)) + 1e-12)


def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def moving_average(x, win):
    win = int(max(3, win))
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x.astype(np.float32), k, mode="same").astype(np.float32)


def detrend_highpass(x, win=151):
    """High-pass / detrend: remove slow baseline so local wiggles show +/- around 0."""
    x = np.asarray(x, dtype=np.float32)
    return (x - moving_average(x, win)).astype(np.float32)


def imshow_with_global_y(ax, img_window, x0, x1, y0, y1, **kwargs):
    extent = [x0, x1, y1 - 1, y0]  # depth increases downward
    return ax.imshow(img_window, extent=extent, aspect="auto", **kwargs)


# -----------------------------
# Wavelet extraction (local stack near y_ref)
# -----------------------------
def extract_wavelet_from_reflector_stack(
    mig_img_T,
    y_ref,
    x_center,
    x_halfwidth=40,
    halfwin=45,
    align=True,
):
    """
    Estimate wavelet by stacking aligned windows around y_ref across neighboring traces.
    mig_img_T: (H depth, W distance)
    Returns (t, w) length = 2*halfwin+1
    """
    H, W = mig_img_T.shape
    y_ref = int(np.clip(y_ref, halfwin, H - halfwin - 1))
    x0 = int(np.clip(x_center - x_halfwidth, 0, W - 1))
    x1 = int(np.clip(x_center + x_halfwidth + 1, 0, W))

    L = 2 * halfwin + 1
    stack = np.zeros(L, dtype=np.float32)
    n_used = 0

    for x in range(x0, x1):
        seg = mig_img_T[y_ref - halfwin : y_ref + halfwin + 1, x].astype(np.float32)
        seg = seg - float(np.mean(seg))  # important (avoid positive-only bias)

        if align:
            k = int(np.argmax(np.abs(seg)))
            seg = np.roll(seg, halfwin - k)

        seg /= (float(np.max(np.abs(seg))) + 1e-12)
        stack += seg
        n_used += 1

    w = stack / max(n_used, 1)

    # center and enforce ~zero-mean wavelet
    k = int(np.argmax(np.abs(w)))
    w = np.roll(w, halfwin - k)
    w = w - float(np.mean(w))

    # taper edges to reduce FFT ringing
    w *= np.hanning(L).astype(np.float32)

    # normalize
    w /= (float(np.max(np.abs(w))) + 1e-12)
    t = np.arange(-halfwin, halfwin + 1)
    return t, w


# -----------------------------
# Deterministic deconvolution
# -----------------------------
def deterministic_decon(trace, wavelet, prewhiten=2e-3):
    """R = IFFT( S * conj(W) / (|W|^2 + eps) ), cropped to trace length."""
    s = np.asarray(trace, dtype=np.float32)
    w = np.asarray(wavelet, dtype=np.float32)

    n = len(s) + len(w) - 1
    nfft = next_pow2(n)

    S = np.fft.fft(s, nfft)
    W = np.fft.fft(np.fft.ifftshift(w), nfft)

    eps = float(prewhiten) * float(np.max(np.abs(W) ** 2) + 1e-12)
    R_full = np.fft.ifft(S * np.conj(W) / (np.abs(W) ** 2 + eps)).real

    mid = (len(w) - 1) // 2
    R = R_full[mid : mid + len(s)].astype(np.float32)

    # remove residual bias
    R = R - float(np.mean(R))
    return R


# -----------------------------
# Reflectivity -> AI (stable)
# -----------------------------
def spike_pick_balanced(R, keep_frac=0.10):
    """Keep top keep_frac tail for positives AND negatives separately."""
    R = np.asarray(R, dtype=np.float32)
    out = np.zeros_like(R)

    pos = R[R > 0]
    neg = -R[R < 0]

    thr_pos = float(np.quantile(pos, 1.0 - keep_frac)) if pos.size >= 5 else None
    thr_neg = float(np.quantile(neg, 1.0 - keep_frac)) if neg.size >= 5 else None

    mask = np.zeros_like(R, dtype=bool)
    if thr_pos is not None:
        mask |= (R >= thr_pos)
    if thr_neg is not None:
        mask |= (-R >= thr_neg)

    if not np.any(mask):  # fallback
        a = np.abs(R)
        thr = float(np.quantile(a, 1.0 - keep_frac)) if a.size else 0.0
        mask = a >= thr

    out[mask] = R[mask]
    return out


def condition_reflectivity_for_ai(R, target_rmax=0.10, q=99.0):
    """Robust scale using percentile(|R|), then clip."""
    R = np.asarray(R, dtype=np.float32).copy()
    scale = float(np.percentile(np.abs(R), q)) + 1e-12
    Rn = R / scale * float(target_rmax)
    Rn = np.clip(Rn, -float(target_rmax), float(target_rmax))
    return Rn.astype(np.float32), scale


def invert_ai_log_stable(R, Z0=9000.0):
    """ln Z = ln Z0 + 2*cumsum(atanh(r))"""
    r = np.asarray(R, dtype=np.float64)
    r = np.clip(r, -0.95, 0.95)

    lnZ0 = np.log(float(Z0))
    ln_ratio = 2.0 * np.arctanh(r)
    lnZ = lnZ0 + np.cumsum(ln_ratio)

    lnZ = np.clip(lnZ, -50.0, 50.0)
    return np.exp(lnZ).astype(np.float32)


# -----------------------------
# Plots
# -----------------------------
def plot_full_trace(trace_norm, mig_img_T, x_trace, y_start, save_path="figure_out/trace_full.svg", show=True):
    """
    Side-by-side plot: migration image (left) with extracted trace location,
    and extracted trace (right). Both panels show y_start to end.
    """
    H, W = mig_img_T.shape
    y_end = H
    y = np.arange(H, dtype=np.int32)

    ensure_dir(save_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: migration image with trace location
    ax0 = axes[0]
    img_window = mig_img_T[y_start:y_end, :]
    im = imshow_with_global_y(ax0, img_window, x0=0, x1=W, y0=y_start, y1=y_end, cmap="gray")
    ax0.axvline(x_trace, color="r", lw=1.5, ls="--", label=f"Extracted trace (x={x_trace})")
    ax0.set_title("Migration image")
    ax0.set_xlabel("Distance (pixel)")
    ax0.set_ylabel("Depth (pixel)")
    ax0.legend(loc="upper right")

    # Right panel: extracted trace
    ax1 = axes[1]
    ax1.plot(trace_norm[y_start:y_end], y[y_start:y_end], lw=1.1)
    ax1.set_title("Extracted trace (normalized)")
    ax1.set_xlabel("Amplitude (z-score)")
    ax1.set_ylabel("Depth (pixel)")
    ax1.set_ylim(y_end - 1, y_start)
    ax1.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_result(trace_hp_full, R_disp, R_spike, AI, y_start, save_path, show=True):
    """Plot trace, reflectivity, and AI from y_start to end."""
    y_end = len(trace_hp_full)
    depth = np.arange(y_start, y_end, dtype=np.int32)
    trace_disp = trace_hp_full[y_start:y_end]
    AI_img = AI[:, None].astype(np.float32)

    ensure_dir(save_path)
    fig, ax = plt.subplots(1, 3, figsize=(16, 6))

    # Trace plot
    ax[0].plot(trace_disp, depth, lw=1.0)
    ax[0].axvline(0, color="k", lw=0.6, ls="--")
    ax[0].set_title("Trace (HP detrended)")
    ax[0].set_xlabel("Amplitude (relative)")
    ax[0].set_ylabel("Depth (pixel)")
    ax[0].set_ylim(y_end - 1, y_start)
    ax[0].grid(True, alpha=0.2)

    # Reflectivity stem plot
    ax[1].axvline(0, color="k", lw=0.8, ls="--")
    nz = np.nonzero(R_spike)[0]
    if nz.size:
        y_stem = depth[nz]
        r_stem = R_spike[nz]
        ax[1].hlines(y_stem, 0.0, r_stem, lw=0.9)
        ax[1].plot(r_stem, y_stem, "o", markersize=2.6)
        mx = float(np.max(np.abs(r_stem)) + 1e-12)
        ax[1].set_xlim(-1.2 * mx, 1.2 * mx)
    ax[1].set_title("Reflectivity (stem)")
    ax[1].set_xlabel("R (signed)")
    ax[1].set_ylabel("Depth (pixel)")
    ax[1].set_ylim(y_end - 1, y_start)
    ax[1].grid(True, alpha=0.2)

    # AI strip
    im = imshow_with_global_y(ax[2], AI_img, x0=0, x1=1, y0=y_start, y1=y_end, cmap="turbo", interpolation="nearest")
    ax[2].set_title("Acoustic Impedance")
    ax[2].set_xlabel("AI strip")
    ax[2].set_ylabel("Depth (pixel)")
    ax[2].set_xticks([])
    cbar = fig.colorbar(im, ax=ax[2], pad=0.02)
    cbar.set_label("AI (relative)")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_ai_overlay(mig_img_T, AI, x_trace, y_start, strip_width=30, save_path="figure_out/ai_overlay.svg", show=True):
    """
    Show migration section (y_start to end) with AI strip embedded.
    """
    H, W = mig_img_T.shape
    y_end = H

    # Calculate strip position centered on x_trace
    x_left = int(np.clip(x_trace - strip_width // 2, 0, W - strip_width))
    x_right = x_left + strip_width

    # Extract migration from y_start to end and convert to RGB
    img_window = mig_img_T[y_start:y_end, :].copy()

    # Normalize migration to 0-1 for grayscale
    mig_min, mig_max = img_window.min(), img_window.max()
    img_norm = (img_window - mig_min) / (mig_max - mig_min + 1e-12)

    # Create RGB image from grayscale migration
    img_rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)

    # Normalize AI to 0-1
    ai_min, ai_max = AI.min(), AI.max()
    ai_norm = (AI - ai_min) / (ai_max - ai_min + 1e-12)

    # Get turbo colormap colors for AI
    cmap = plt.cm.turbo
    ai_colors = cmap(ai_norm)[:, :3]  # (len(AI), 3) RGB values

    # Insert AI strip into the RGB image
    for i in range(x_left, x_right):
        img_rgb[:, i, :] = ai_colors

    ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(14, 6))

    # Show combined image
    extent = [0, W, y_end - 1, y_start]
    ax.imshow(img_rgb, extent=extent, aspect="auto")

    ax.set_title("Migration with AI overlay")
    ax.set_xlabel("Distance (pixel)")
    ax.set_ylabel("Depth (pixel)")

    # Add colorbar for AI reference
    sm = plt.cm.ScalarMappable(cmap='turbo', norm=plt.Normalize(vmin=ai_min, vmax=ai_max))
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label("Acoustic Impedance (relative)")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    if show:
        plt.show()
    else:
        plt.close(fig)


def analyze_ai(
    mig_img_T,
    trace_hp_full,
    x_trace,
    y_start,
    wavelet_halfwin=45,
    wavelet_x_halfwidth=40,
    prewhiten=2e-3,
    reflect_detrend_win=31,
    stem_keep_frac=0.10,
    ai_target_rmax=0.10,
    ai_q=99.0,
    Z0=9000.0,
):
    """Analyze AI from y_start to end."""
    H, W = mig_img_T.shape
    y_end = H
    y_ref = (y_start + y_end) // 2

    # local wavelet centered at y_ref
    tw, w = extract_wavelet_from_reflector_stack(
        mig_img_T=mig_img_T,
        y_ref=y_ref,
        x_center=x_trace,
        x_halfwidth=wavelet_x_halfwidth,
        halfwin=wavelet_halfwin,
        align=True,
    )

    # wavelet plot
    wavelet_path = "figure_out/wavelet.svg"
    ensure_dir(wavelet_path)
    plt.figure(figsize=(6, 3))
    plt.plot(tw, w, lw=1.5)
    plt.axvline(0, color="k", lw=0.8, ls="--")
    plt.title("Estimated wavelet (local stack)")
    plt.xlabel("Samples (relative)")
    plt.ylabel("Amplitude (norm.)")
    plt.tight_layout()
    plt.savefig(wavelet_path, bbox_inches="tight", pad_inches=0.05)
    plt.show()

    # decon on full trace (HP-normalized)
    R_full = deterministic_decon(trace_hp_full, w, prewhiten=prewhiten)
    R = R_full[y_start:y_end]

    print(f"\n[Analysis y={y_start}:{y_end}] y_ref={y_ref}")
    print("R min/max:", float(np.min(R)), float(np.max(R)))
    print("R mean:", float(np.mean(R)))

    # detrend reflectivity for meaningful +/- stems
    R_disp = detrend_highpass(R, win=reflect_detrend_win)
    print("R_disp min/max:", float(np.min(R_disp)), float(np.max(R_disp)))
    print("R_disp negatives:", int(np.sum(R_disp < 0)), "of", R_disp.size)

    R_spike = spike_pick_balanced(R_disp, keep_frac=stem_keep_frac)
    print("R_spike min/max:", float(np.min(R_spike)), float(np.max(R_spike)))

    R_ai, scale = condition_reflectivity_for_ai(R_disp, target_rmax=ai_target_rmax, q=ai_q)
    print(f"AI conditioning: |R| p{ai_q} scale={scale:.5g}, target_rmax={ai_target_rmax}")
    print("R_ai min/max:", float(np.min(R_ai)), float(np.max(R_ai)))

    AI = invert_ai_log_stable(R_ai, Z0=Z0)

    out_path = "figure_out/result.svg"
    plot_result(trace_hp_full, R_disp, R_spike, AI, y_start, out_path, show=True)

    # Overlay AI on migration section
    overlay_path = "figure_out/ai_overlay.svg"
    plot_ai_overlay(mig_img_T, AI, x_trace, y_start, strip_width=AI_STRIP_WIDTH, save_path=overlay_path, show=True)


# -----------------------------
# Main
# -----------------------------
def main():
    raw = np.load(MIGRATION_NPY).astype(np.float32)
    raw_clip = raw[:, TOP_CLIP_COLS:]     # (Nx, Nz)
    mig_img_T = raw_clip.T                # (Nz, Nx)
    Nx, Nz = raw_clip.shape

    x_trace = int(np.clip(X_TRACE, 0, Nx - 1))

    # Signed raw trace from NPY
    trace_raw_full = raw_clip[x_trace, :].copy()
    print("RAW trace min/max:", float(np.min(trace_raw_full)), float(np.max(trace_raw_full)))
    print("RAW trace negatives:", int(np.sum(trace_raw_full < 0)), "of", trace_raw_full.size)

    # Normalize only (then HP for wiggle/decon)
    trace_norm_full = zscore(trace_raw_full)

    plot_full_trace(trace_norm_full, mig_img_T, x_trace, Y_START, save_path="figure_out/trace_full.svg", show=True)

    # HP version fixes "wiggle only negative in window"
    trace_hp_full = detrend_highpass(trace_norm_full, win=TRACE_HP_WIN)
    print("HP trace min/max:", float(np.min(trace_hp_full)), float(np.max(trace_hp_full)))

    analyze_ai(
        mig_img_T=mig_img_T,
        trace_hp_full=trace_hp_full,
        x_trace=x_trace,
        y_start=Y_START,
        wavelet_halfwin=WAVELET_HALFWIN,
        wavelet_x_halfwidth=WAVELET_X_HALFWIDTH,
        prewhiten=PREWHITEN,
        reflect_detrend_win=REFLECT_DETREND_WIN,
        stem_keep_frac=STEM_KEEP_FRAC,
        ai_target_rmax=AI_TARGET_RMAX,
        ai_q=AI_Q,
        Z0=Z0,
    )


if __name__ == "__main__":
    main()
