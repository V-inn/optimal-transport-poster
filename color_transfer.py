"""
color_transfer.py -- OT between the colour distributions of two photos (poster, block 3).

    python color_transfer.py                         # data/ferret.png, data/octopus.png
    python color_transfer.py photo.png palette.png   # any pair

Each photo's pixel colours form a distribution in the RGB cube (a 32^3 colour histogram).
Exact OT (POT's ot.emd, cost = squared colour distance) moves one histogram onto the other;
each colour bin is sent to the average of the colours it is matched with (barycentric map),
and every pixel is shifted by that displacement (smoothed over the colour cube), so the
texture of the photo is kept and smooth gradients stay smooth.

Outputs
  out/color/<a>_with_<b>_colors.png, out/color/<b>_with_<a>_colors.png   recoloured photos
  out/color/overview.png                                               both directions
  figures/fig_color_ot_mse.png                                         the poster figure
"""
import os
import sys

import numpy as np
import ot
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out", "color")
LEVELS = 32


def histogram(rgb, levels=LEVELS):
    """Occupied colour bins: mean colour (n, 3), weight (n,), bin index of every pixel."""
    idx = np.minimum((rgb * levels).astype(np.int64), levels - 1)
    flat = (idx[..., 0] * levels + idx[..., 1]) * levels + idx[..., 2]
    bins, inverse, counts = np.unique(flat.ravel(), return_inverse=True, return_counts=True)
    sums = np.zeros((len(bins), 3))
    np.add.at(sums, inverse, rgb.reshape(-1, 3))
    return sums / counts[:, None], counts / counts.sum(), inverse.reshape(rgb.shape[:2])


def transfer(src, pal, smooth=1.5, levels=LEVELS):
    """src recoloured with the colour distribution of pal (both float RGB in [0, 1]).

    smooth > 0: the displacement T(x) - x is smoothed over the colour cube with a Gaussian of
    `smooth` bins (then interpolated at every pixel colour).  The exact OT map jumps between
    neighbouring bins, which shows up as contour bands in smooth gradients (blurred
    backgrounds, sky); smoothing the map removes them (regularized OT colour transfer)."""
    from scipy.ndimage import gaussian_filter, map_coordinates
    xs, a, pix_bin = histogram(src, levels)
    yt, b, _ = histogram(pal, levels)
    P = ot.emd(a, b, ot.dist(xs, yt), numItermax=10_000_000)
    T = (P @ yt) / a[:, None]                         # where each colour bin is sent
    if not smooth:
        return np.clip(src + (T - xs)[pix_bin], 0, 1)  # move every pixel with its bin
    idx = tuple(np.minimum((xs * levels).astype(np.int64), levels - 1).T)
    W = np.zeros((levels,) * 3)
    D = np.zeros((levels,) * 3 + (3,))
    np.add.at(W, idx, a)
    np.add.at(D, idx, a[:, None] * (T - xs))
    Ws = gaussian_filter(W, smooth, mode="nearest")
    field = [gaussian_filter(D[..., c], smooth, mode="nearest") / np.maximum(Ws, 1e-12)
             for c in range(3)]
    coords = (src.reshape(-1, 3) * levels - 0.5).T     # bin centres sit at (i + 0.5) / levels
    disp = np.stack([map_coordinates(f, coords, order=1, mode="nearest") for f in field], 1)
    return np.clip(src + disp.reshape(src.shape), 0, 1)


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), np.float64) / 255


def square(img, px=600):
    """Centre crop to a square and resize (so two photos can be averaged pixel by pixel)."""
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    im = Image.fromarray((img[y0:y0 + s, x0:x0 + s] * 255).round().astype(np.uint8))
    return np.asarray(im.resize((px, px), Image.LANCZOS), np.float64) / 255


def color_cloud_axes(A, B, n=6000, seed=0):
    """Same random pixels for every panel, and a 2D view of RGB: the two principal
    directions of the colours of both photos (roughly brightness and hue)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(A.shape[0] * A.shape[1], n, replace=False)
    both = np.concatenate([A.reshape(-1, 3), B.reshape(-1, 3)])
    mean = both.mean(0)
    _, _, Vt = np.linalg.svd(both[rng.choice(len(both), 20000, replace=False)] - mean,
                             full_matrices=False)
    return idx, mean, Vt[:2]


def overview(rows, path):
    """photo | palette | OT recoloring, one row per direction."""
    import make_figures as mf
    import matplotlib.pyplot as plt
    W, gap, row_h, top, vgap = 1800, 24, 420, 60, 30
    H = top + len(rows) * row_h + (len(rows) - 1) * vgap + 10
    fig = mf.fig_pt(W, H)
    titles = ["photo", "palette", "OT recoloring"]
    for r, imgs in enumerate(rows):
        widths = [row_h * im.shape[1] / im.shape[0] for im in imgs]
        scale = min(1.0, (W - 2 * gap) / sum(widths))
        hh = row_h * scale
        x = (W - scale * sum(widths) - 2 * gap) / 2
        y = H - top - (r + 1) * row_h - r * vgap + (row_h - hh) / 2
        for k, (im, w) in enumerate(zip(imgs, widths)):
            ax = fig.add_axes([x / W, y / H, w * scale / W, hh / H])
            ax.imshow(im, interpolation="lanczos"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if k == 2:
                mf.frame(ax)
            if r == 0:
                ax.set_title(titles[k], fontsize=mf.FS, pad=10)
            x += w * scale + gap
    fig.savefig(path, dpi=150); plt.close(fig)
    Image.open(path).convert("RGB").save(path)


def poster_figure(photo, palette, recolored, path, cloud_h=0.58):
    """Barycenters with weights (1-t, t), t = 0, 1/2, 1, half-block width, three rows:
       OT    : colours moved along the OT map,   (1-t) x + t T(x)   (the shape stays)
       colors: the same pixels as points of the RGB cube (2D projection), for the OT row
       MSE   : pixel average of the two photos,  (1-t) A + t B      (the photos superpose)"""
    import make_figures as mf
    import matplotlib.pyplot as plt
    A, B, TA = square(photo), square(palette), square(recolored)
    ts = [0.0, 0.5, 1.0]
    ot_imgs = [(1 - t) * A + t * TA for t in ts]
    idx, mean, V = color_cloud_axes(A, B)
    clouds = [im.reshape(-1, 3)[idx] for im in ot_imgs]
    proj = [(c - mean) @ V.T for c in clouds]
    allp = np.concatenate(proj + [(B.reshape(-1, 3)[idx] - mean) @ V.T])
    lo, hi = np.percentile(allp, 0.5, axis=0), np.percentile(allp, 99.5, axis=0)
    pad = 0.06 * (hi - lo)
    W = mf.HALF_W
    lab_w = 70
    cell = (W - lab_w) / 3
    ch = cell * cloud_h
    H = 2 * cell + ch + 50
    fig = mf.fig_pt(W, H)
    rows = [("OT", mf.ACC, "img", ot_imgs, cell),
            ("colors", mf.ACC, "cloud", proj, ch),
            ("MSE", mf.GRAY, "img", [(1 - t) * A + t * B for t in ts], cell)]
    y_top = H
    for r, (lab, col, kind, items, h) in enumerate(rows):
        y0 = y_top - h
        for k, it in enumerate(items):
            ax = fig.add_axes([(lab_w + k * cell + 5) / W, (y0 + 5) / H, (cell - 10) / W, (h - 10) / H])
            ax.set_xticks([]); ax.set_yticks([])
            if kind == "img":
                ax.imshow(np.clip(it, 0, 1), interpolation="lanczos")
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if k == 1:
                    mf.frame(ax, col, 3)
            else:
                ax.set_facecolor("#262626")
                ax.scatter(it[:, 0], it[:, 1], c=np.clip(clouds[k], 0, 1), s=6, lw=0,
                           rasterized=True)
                ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_color(mf.GRAY); sp.set_linewidth(1)
            if r == len(rows) - 1:
                ax.text(0.5, -0.03, ["$t=0$", r"$t=\frac{1}{2}$", "$t=1$"][k], transform=ax.transAxes,
                        ha="center", va="top", fontsize=mf.FS - 4, color=mf.GRAY)
        fig.text(8 / W, (y0 + h / 2) / H, lab, fontsize=(mf.FS + 2 if kind == "img" else mf.FS - 4),
                 va="center", ha="left", color=col, fontweight="bold", rotation=90)
        y_top = y0
    fig.savefig(path, dpi=200); plt.close(fig)
    Image.open(path).convert("RGB").save(path)
    print("saved", os.path.relpath(path, HERE))


def main(pa, pb):
    os.makedirs(OUT, exist_ok=True)
    A, B = load(pa), load(pb)
    na, nb = (os.path.splitext(os.path.basename(p))[0] for p in (pa, pb))
    AB, BA = transfer(A, B), transfer(B, A)
    for img, name in [(AB, f"{na}_with_{nb}_colors.png"), (BA, f"{nb}_with_{na}_colors.png")]:
        Image.fromarray((img * 255).round().astype(np.uint8)).save(os.path.join(OUT, name))
        print("saved", os.path.relpath(os.path.join(OUT, name), HERE))
    overview([(A, B, AB), (B, A, BA)], os.path.join(OUT, "overview.png"))
    print("saved", os.path.relpath(os.path.join(OUT, "overview.png"), HERE))
    poster_figure(A, B, AB, os.path.join(HERE, "figures", "fig_color_ot_mse.png"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, "ferret.png"),
         sys.argv[2] if len(sys.argv) > 2 else os.path.join(DATA, "octopus.png"))
