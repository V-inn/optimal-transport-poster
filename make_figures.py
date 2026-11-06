"""
make_figures.py -- the three "drawn" figures of the poster, and the shared style.

    python make_figures.py              # all three
    python make_figures.py rd avg       # only some of them

  rd      figures/fig_rd_barycenter.pdf   block 1: weighted points and their barycenter
  points  figures/fig_points.pdf          block 2: optimal coupling, Gaussian cloud -> Cristo
  avg     figures/fig_mse_average.pdf     block 3: average of two bumps, MSE vs W2

The other two figures come from color_transfer.py (fig_color_ot_mse) and
barycenter_triangle.py (fig_triangle), which import the style defined here.

Every figure is drawn at its real size on the poster (in points, 1/72 in) on the poster
background, so the fonts match the LaTeX text.  Restyle everything from the CONFIG block.
"""
from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
logging.getLogger("fontTools").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import ot

from otlib import load_shape, sinkhorn_log

# =============================================================================
# CONFIG
# =============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "data")        # input images (see README: not in the repo)
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

PT = 1 / 72                    # one poster point, in inches
FULL_W = 1915                  # width of a full-width block on the poster (pt)
HALF_W = 900                   # width of one half of a block (pt)

BG = "#000000"                 # impaTeX dark mode
FG = "#F3F3F3"                 # impaTech white
ACC = "#FFC555"                # impaTech orange (main accent)
GRAY = "#8C8C8C"
CMAP = LinearSegmentedColormap.from_list(   # black -> orange -> cream, for densities
    "impa_orange", [BG, "#3A2400", "#A36A12", ACC, "#FFF2D2"])

FS = 30                        # base font size inside the figures (pt)
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": GRAY,
    "xtick.color": FG, "ytick.color": FG,
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "cm",
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS - 6, "ytick.labelsize": FS - 6, "legend.fontsize": FS - 6,
    "axes.linewidth": 2, "lines.linewidth": 5,
    "xtick.major.width": 2, "ytick.major.width": 2,
    "xtick.major.size": 8, "ytick.major.size": 8,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "image.interpolation": "bilinear",
})


def fig_pt(w, h):
    return plt.figure(figsize=(w * PT, h * PT))


def save(fig, name, png=False, dpi=200):
    path = os.path.join(OUT, name + (".png" if png else ".pdf"))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    if png:                                    # drop the alpha channel (safer for TV players)
        from PIL import Image
        Image.open(path).convert("RGB").save(path)
    print("  saved", os.path.relpath(path, HERE))


def crop_box(fields, pad=8):
    """Square crop around the union of the supports (makes small shapes bigger)."""
    m = np.zeros_like(fields[0], dtype=bool)
    for f in fields:
        m |= f > 1e-3 * f.max()
    r, c = np.nonzero(m)
    d = fields[0].shape[0]
    half = max(r.max() - r.min(), c.max() - c.min()) // 2 + pad
    rc, cc = (r.max() + r.min()) // 2, (c.max() + c.min()) // 2
    r0 = int(np.clip(rc - half, 0, d - 2 * half)); c0 = int(np.clip(cc - half, 0, d - 2 * half))
    return (slice(r0, r0 + 2 * half), slice(c0, c0 + 2 * half))


def frame(ax, color=ACC, lw=3):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(color); sp.set_linewidth(lw)


def gauss(x, m, s):
    g = np.exp(-(x - m) ** 2 / (2 * s * s))
    return g / (g.sum() * (x[1] - x[0]))


# =============================================================================
# Block 1 -- the barycenter of points in R^d
# =============================================================================
def rd_barycenter():
    """Weighted points x_i (size ~ lambda_i) and their barycenter sum_i lambda_i x_i."""
    pts = np.array([[0.10, 0.62], [0.32, 0.12], [0.55, 0.88], [0.80, 0.30], [0.95, 0.75]])
    lam = np.array([0.30, 0.15, 0.20, 0.25, 0.10])
    bar = lam @ pts
    W, H = HALF_W, 330
    fig = fig_pt(W, H)
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.92])
    for (x, y), l in zip(pts, lam):
        ax.plot([bar[0], x], [bar[1], y], color=GRAY, lw=2.5, ls=(0, (4, 3)), zorder=1)
        ax.scatter([x], [y], s=4200 * l, color=FG, lw=0, zorder=2)
    ax.scatter(*bar, s=520, color=ACC, marker="*", lw=0, zorder=3)
    ax.text(bar[0] + 0.03, bar[1] + 0.03, r"$\bar x$", color=ACC, fontsize=FS + 8)
    ax.text(pts[0, 0] + 0.045, pts[0, 1] + 0.03, r"$x_i$", fontsize=FS + 2)
    ax.set_xlim(0.0, 1.05); ax.set_ylim(0.0, 1.0)
    ax.set_aspect("auto"); ax.axis("off")
    save(fig, "fig_rd_barycenter")


# =============================================================================
# Block 2 -- optimal coupling between two point clouds
# =============================================================================
def _sample_from_field(f, n, rng):
    idx = rng.choice(f.size, size=n, replace=False, p=f.ravel() / f.sum())
    r, c = np.unravel_index(idx, f.shape)
    d = f.shape[0]
    pts = np.stack([c / d, 1 - r / d], 1)
    return pts + rng.uniform(-0.5, 0.5, pts.shape) / d


def point_clouds(both=False, grid=44, seed=3):
    """Exact OT plan (an assignment) from a Gaussian cloud to the Cristo Redentor,
    drawn as pixel art (one target point per pixel of a grid x grid silhouette).
    both=True adds the entropic plan next to it (extra_points_eps)."""
    rng = np.random.default_rng(seed)
    W, H = ((2 * HALF_W + 100) / 2, 560) if both else (HALF_W, 470)
    xmax = W / H                                 # data units: the panel is xmax x 1
    f = load_shape(os.path.join(IMG_DIR, "cristo_redentor.jpg"), grid, margin=0.0)
    r, c = np.nonzero(f > 0)
    tgt = np.stack([c / grid, 1 - r / grid], 1)
    n = len(tgt)
    sx = 0.22 * xmax
    src = rng.normal([sx, 0.5], [0.05, 0.11], size=(n, 2))
    lo, hi = tgt.min(0), tgt.max(0)
    tgt = (tgt - lo) / (hi - lo).max() * 0.9
    tgt += np.array([xmax - 0.04 - tgt[:, 0].max(), 0.5 - 0.5 * (tgt[:, 1].max() + tgt[:, 1].min())])
    a = b = np.ones(n) / n
    C = ot.dist(src, tgt)                                     # squared Euclidean cost
    plans = [(None if not both else r"$\varepsilon=0$: one arrow per point", ot.emd(a, b, C))]
    if both:
        plans.append((r"$\varepsilon>0$: mass is spread", sinkhorn_log(a, b, C, 0.01)[0]))
    fig = fig_pt(W * len(plans), H)
    gs = fig.add_gridspec(1, len(plans), left=0.01, right=0.99, bottom=0.01,
                          top=0.88 if both else 0.99, wspace=0.05)
    for k, (title, P) in enumerate(plans):
        ax = fig.add_subplot(gs[k])
        Pm = P / P.max()
        I, J = np.nonzero(Pm > 0.03)
        for i, j in zip(I, J):
            ax.plot([src[i, 0], tgt[j, 0]], [src[i, 1], tgt[j, 1]], color=GRAY,
                    lw=0.9, alpha=0.3 * Pm[i, j], solid_capstyle="round")
        ax.scatter(*src.T, s=16, color=FG, zorder=3, lw=0)
        ax.scatter(*tgt.T, s=22, color=ACC, zorder=3, lw=0, marker="s")
        ax.text(sx, 0.08, r"$\mu$", ha="center", fontsize=FS + 8, color=FG)
        ax.text(tgt[:, 0].min() - 0.06, 0.88, r"$\nu$", ha="center", fontsize=FS + 8, color=ACC)
        if title:
            ax.set_title(title, fontsize=FS - 2)
        ax.set_xlim(0, xmax); ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal"); ax.axis("off")
    save(fig, "extra_points_eps" if both else "fig_points")


# =============================================================================
# Block 3 -- average of two bumps: MSE vs Wasserstein
# =============================================================================
def mse_average():
    """Average of two bumps: MSE mean (two half-bumps) vs W2 barycenter (one bump in between)."""
    x = np.linspace(-3, 9, 1200)
    s, t = 0.7, 6.0
    mu, nu = gauss(x, 0, s), gauss(x, t, s)
    fig = fig_pt(HALF_W, 300)
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.78])
    ax.fill_between(x, 0.5 * (mu + nu), color=GRAY, alpha=0.6, lw=0,
                    label=r"$\frac{1}{2}\mu_1+\frac{1}{2}\mu_2$ (MSE)")
    ax.fill_between(x, gauss(x, t / 2, s), color=ACC, alpha=0.95, lw=0, label=r"$\bar\nu$ ($W_2$)")
    ax.plot(x, mu, color=FG, lw=2.5, ls="--"); ax.plot(x, nu, color=FG, lw=2.5, ls="--")
    ax.text(0, 0.62, r"$\mu_1$", ha="center", fontsize=FS + 2)
    ax.text(t, 0.62, r"$\mu_2$", ha="center", fontsize=FS + 2)
    ax.set_yticks([]); ax.set_xticks([]); ax.set_ylim(0, 0.75)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper center", fontsize=FS - 4, ncol=2, bbox_to_anchor=(0.5, 1.2),
              columnspacing=1.2, handlelength=1.2)
    save(fig, "fig_mse_average")


FIGS = {"rd": rd_barycenter, "points": point_clouds, "avg": mse_average}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(FIGS)
    for name in todo:
        t0 = time.time()
        print(f"[{name}]")
        FIGS[name]()
        print(f"  {time.time() - t0:.1f}s")
