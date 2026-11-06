"""
barycenter_triangle.py -- Wasserstein barycenter of THREE photos (cat, snowy owl, brown
owl) for weights on a grid of the triangle: figures/fig_triangle.png (poster, block 4).

    python barycenter_triangle.py              # 192 px, eps 1.5e-4, debiased, weights in thirds
    python barycenter_triangle.py --n-sub 4    # weights in quarters (15 pictures)
    python barycenter_triangle.py --size 384 --eps 3.5e-5   # sharper (GPU recommended)
    python barycenter_triangle.py --device cpu # force the CPU (default: CUDA GPU if any)

On a CUDA GPU the barycenters are computed one after another (each one batches its
photos through the kernel); on the CPU they run in parallel processes (--workers).

Inputs
  cat        data/cat_photo.png (segmented and placed exactly as in photo_barycenter.py)
  snowy owl  first tile of data/snowy_to_brown_strip.png   (strip.png made by
             `python meowl_gpu.py --pair snowy_owl brown_owl --steps 11`)
  brown owl  last tile of the same strip
  (the strip tiles are already on the same canvas as the cat: animals at 86% of the
   height, eye midpoint at (0.5, 0.19))
Outputs
  out/triangle/*.npz      one barycenter per weight vector (reused on the next run)
  figures/fig_triangle.png   the poster figure (half-block width)
"""
import argparse
import os
import time

import photo_barycenter as ml      # sets one thread per process, then imports numpy
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage as ndi

import otlib

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CACHE = os.path.join(HERE, "out", "triangle")
NAMES = ["cat", "snowy owl", "brown owl"]


def inputs(N):
    """mass, colour for the three animals on the N x N grid."""
    out = [ml.aligned("cat", N)]
    strip = np.asarray(Image.open(os.path.join(DATA, "snowy_to_brown_strip.png")).convert("RGB"),
                       np.float32) / 255
    tile = strip.shape[0]
    for i in (0, strip.shape[1] // tile - 1):            # first and last tile
        t = cv2.resize(strip[:, tile * i:tile * (i + 1)], (N, N), interpolation=cv2.INTER_AREA)
        mask = ndi.binary_fill_holes(t.max(2) > 0.04)
        lab, n = ndi.label(mask)
        mask = lab == (1 + np.argmax(ndi.sum(mask, lab, range(1, n + 1))))
        out.append((mask / mask.sum(), np.clip(t, 0, 1)))
    return out


def weights(n_sub):
    return [np.array([i, j, n_sub - i - j]) / n_sub
            for i in range(n_sub + 1) for j in range(n_sub + 1 - i)]


def key_of(lam):
    return "_".join(f"{x:.3f}" for x in lam)


def cache_path(args, lam):
    return os.path.join(CACHE, f"{key_of(lam)}_{args.size}_{args.eps:g}.npz")


def job(job_args):
    args, lam, ins = job_args
    key = key_of(lam)
    t0 = time.time()
    use = [k for k in range(3) if lam[k] > 0]            # zero weight: leave that photo out
    if len(use) == 1:
        m, c = ins[use[0]]
        out = dict(mass=m, colour=c)
    else:
        fields = [ins[k][0] for k in use]
        if not args.quiet:
            print(f"  lambda = {key}: computing ({len(use)} photos, {args.size} px, eps {args.eps:g})",
                  flush=True)
        b, lnu, lnKtu, K = ml.barycenter(fields, lam[use], args.eps, debias=True,
                                         device=args.device, dtype=args.dtype,
                                         verbose=not args.quiet)
        out = dict(mass=b, colour=ml.transported_colour([ins[k][1] for k in use], lam[use] / lam[use].sum(),
                                                        lnu, lnKtu, K))
    np.savez(cache_path(args, lam), **out)
    print(f"  lambda = {key} done in {time.time() - t0:.0f}s", flush=True)
    return key, out


def figure(frames, n_sub, size):
    import make_figures as mf
    W = mf.HALF_W
    V = np.array([[0.5, np.sqrt(3) / 2], [0.0, 0.0], [1.0, 0.0]])   # cat top, owls at the base
    cell = 1.0 / (n_sub + 0.55)
    side = W / (1 + cell)
    img = cell * side * 0.98
    lab = 50
    H = lab + side * np.sqrt(3) / 2 + img + 6           # top picture fully inside
    fig = mf.fig_pt(W, H)

    def centre(p):
        return 0.5 * cell * side + p[0] * side, lab + 0.5 * img + p[1] * side

    # crop all pictures to the same square around the union of the animals
    box = mf.crop_box([f["mass"] for f in frames.values()], pad=4)
    for lam in weights(n_sub):
        f = frames["_".join(f"{x:.3f}" for x in lam)]
        m = f["mass"]
        if lam.max() < 1:
            # the debiased barycenter rings (thin stripes) where little mass is left, e.g.
            # where only one animal has a tail; a 1-pixel blur of the opacity hides it
            # (the kernel itself is sqrt(eps)*size ~ 2.4 pixels wide)
            m = ndi.gaussian_filter(m, 1.0)
        alpha = np.clip(m / np.quantile(m[m > 0], 0.97), 0, 1)[..., None]
        rgb = (f["colour"] * alpha)[box]
        cx, cy = centre(lam @ V)
        ax = fig.add_axes([(cx - img / 2) / W, (cy - img / 2) / H, img / W, img / H])
        ax.imshow(np.clip(rgb, 0, 1), interpolation="bicubic")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if lam.max() == 1:
            mf.frame(ax)
        if np.allclose(lam, 1 / 3):
            mf.frame(ax, mf.FG, 2)
    cx, cy = centre(V[0])
    fig.text((cx + 0.58 * img) / W, cy / H, NAMES[0], fontsize=mf.FS, va="center")
    for v, text, ha, sgn in zip(V[1:], NAMES[1:], ["left", "right"], [-1, 1]):
        cx, cy = centre(v)
        fig.text((cx + sgn * img / 2) / W, (cy - 0.52 * img) / H, text, fontsize=mf.FS - 2,
                 ha=ha, va="top")
    mf.save(fig, "fig_triangle", png=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", type=int, default=192)
    p.add_argument("--eps", type=float, default=1.5e-4,
                   help="keep eps * size^2 >= ~5: below that the fast CPU kernel leaves a dot pattern")
    p.add_argument("--n-sub", type=int, default=3, help="weights in steps of 1/n-sub")
    p.add_argument("--workers", type=int, default=os.cpu_count(),
                   help="parallel processes (CPU only; on the GPU one barycenter after another)")
    p.add_argument("--quiet", action="store_true", help="no progress lines while computing")
    otlib.add_device_args(p)
    args = p.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    lams = sorted(weights(args.n_sub), key=lambda l: -np.count_nonzero(l))   # 3-way first
    t0 = time.time()
    frames = {key_of(l): dict(np.load(cache_path(args, l))) for l in lams
              if os.path.exists(cache_path(args, l))}
    todo = [l for l in lams if key_of(l) not in frames]
    if todo:
        dev = otlib.pick_device(args.device)
        args.device = dev.type
        if dev.type == "cuda" and args.dtype == "auto":
            print(f"device: {otlib.describe(dev, otlib.pick_dtype(dev, args.dtype))} for eps < 1e-4, "
                  f"float64 matrix products for eps >= 1e-4", flush=True)
        else:
            print(f"device: {otlib.describe(dev, otlib.pick_dtype(dev, args.dtype))}", flush=True)
        print(f"{len(todo)} barycenters to compute ({len(lams) - len(todo)} already in {CACHE})",
              flush=True)
        ins = inputs(args.size)                          # segmented once, shared by all jobs
        jobs = [(args, l, ins) for l in todo]
        if dev.type == "cuda" or min(args.workers, len(jobs)) <= 1:
            frames.update(job(j) for j in jobs)          # one GPU: one barycenter after another
        else:
            from multiprocessing import Pool
            with Pool(min(args.workers, len(jobs))) as pool:
                frames.update(pool.imap_unordered(job, jobs))
    print(f"barycenters done in {time.time() - t0:.0f}s")
    figure(frames, args.n_sub, args.size)


if __name__ == "__main__":
    main()
