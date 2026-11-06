"""
photo_barycenter.py -- Wasserstein barycenter of two photos (cat + owl = "meowl"), on a CUDA
GPU (PyTorch) or, without one, on the CPU.

Also the library used by barycenter_triangle.py and meowl_gpu.py (segmentation, alignment,
debiased Sinkhorn barycenter, colours carried along the transport plan).

Why the first meowl was blurry
------------------------------
Not because of too few iterations (those runs had converged).  Two causes:
  1. Entropy: the Sinkhorn kernel exp(-|x-y|^2 / eps) smooths over ~sqrt(eps).
     -> use a smaller eps (needs a finer grid and more iterations).
  2. The Sinkhorn barycenter is biased towards blurry shapes (even the barycenter
     of ONE image with itself comes out blurred).
     -> use the *debiased* barycenter (Janati, Cuturi & Gramfort, ICML 2020):
        one extra vector c with c * (K c) = b removes that bias.
To make small eps affordable the script also uses eps-scaling: it solves for a
large eps first and shrinks it, reusing the dual potentials.

Usage
-----
    pip install -r requirements.txt                  # (PyTorch with CUDA: see README.md)
    python photo_barycenter.py                       # defaults: 256 px, eps 1e-4, debiased
    python photo_barycenter.py --size 384 --eps 4e-5 # sharper, slower (exact kernel)
    python photo_barycenter.py --steps 8             # 9 frames (t = k/8) for a smoother GIF
    python photo_barycenter.py --no-debias           # classical Sinkhorn barycenter
    python photo_barycenter.py --device cpu          # force the CPU (default: GPU if any)

Rule of thumb: keep eps >= 2/size^2 (kernel at least ~1 pixel wide).
GPU (default when PyTorch sees CUDA): float32, exact log-sum-exp kernel at every eps
(size^3 work per step, but massively parallel); frames are computed one after another.
CPU: float64 as before.  Down to eps = 1e-4 a fast matrix-product kernel is used (a few
minutes per frame on one core); below that it switches to the exact log-sum-exp kernel,
which is stable but costs size^3 per step (expect tens of minutes per frame at 256 px).
On the CPU frames run in parallel processes (--workers).

Inputs: data/cat_photo.png and data/owl_photo.png (edit PHOTOS for others).
Outputs (folder out/meowl/): meowl.png (t = 1/2), strip.png, meowl.gif, frames/*.npz
"""
import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")       # CPU: frames run in parallel processes
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
from scipy import ndimage as ndi
from scipy.special import logsumexp
from PIL import Image

import otlib

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

PHOTOS = {
    # file, GrabCut rectangle (x, y, w, h) around the animal, eye centres (x, y) in pixels
    "cat": ("cat_photo.png", (55, 30, 450, 580), ((254, 141), (325, 146))),
    "owl": ("owl_photo.png", (55, 65, 190, 385), ((137, 124), (184, 123))),
}
WORK = 1024             # resolution of the aligned canvas before downsampling


# =============================================================================
# Images -> mass + colour on an N x N grid
# =============================================================================
def segment(name):
    import cv2
    fname, rect, _ = PHOTOS[name]
    img = np.asarray(Image.open(os.path.join(DATA, fname)).convert("RGB"))
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.grabCut(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), m, rect, np.zeros((1, 65)),
                np.zeros((1, 65)), 8, cv2.GC_INIT_WITH_RECT)
    raw = (m == 1) | (m == 3)
    if name == "cat":                                   # cut the white halo of the background
        mask = ndi.binary_erosion(raw, iterations=3)
    else:                                               # owl: remove twigs, bridge the branch
        mask = ndi.binary_opening(raw, structure=np.ones((5, 5)))
        mask = ndi.binary_closing(mask, structure=np.ones((45, 3)))
    mask = ndi.binary_fill_holes(mask)
    lab, n = ndi.label(mask)
    mask = lab == (1 + np.argmax(ndi.sum(mask, lab, range(1, n + 1))))
    added = mask & ~raw
    if added.any():                                     # repaint where the branch was
        img = cv2.inpaint(img, (added * 255).astype(np.uint8), 7, cv2.INPAINT_TELEA)
    return img, mask


def aligned(name, N):
    """Animal scaled to 86% of the canvas height, eye midpoint at (0.5, 0.19)."""
    import cv2
    img, mask = segment(name)
    (x1, y1), (x2, y2) = PHOTOS[name][2]
    rows = np.where(mask.any(1))[0]
    s = 0.86 * WORK / (rows.max() - rows.min())
    A = np.float32([[s, 0, WORK / 2 - s * (x1 + x2) / 2], [0, s, 0.19 * WORK - s * (y1 + y2) / 2]])
    warp = lambda x: cv2.warpAffine(x, A, (WORK, WORK), flags=cv2.INTER_AREA)
    m = warp(mask.astype(np.float32))
    rgb = warp(img.astype(np.float32) / 255)
    down = lambda x: cv2.resize(x, (N, N), interpolation=cv2.INTER_AREA)
    mass = down(m)
    col = down(rgb * m[..., None]) / np.maximum(mass[..., None], 1e-6)
    return mass / mass.sum(), np.clip(col, 0, 1)


# =============================================================================
# Log-domain convolutional Sinkhorn (Gibbs kernel exp(-|x-y|^2/eps) on [0,1]^2)
# =============================================================================
# The kernel is otlib.LogGibbs (PyTorch, CPU or CUDA).  K = K1 (x) K1 is separable, so
# log(K exp(V)) is two log-sum-exp passes, one along each axis of the grid, done for all
# the fields at once.  In float64 (CPU default) it is the fast matrix-product kernel down
# to eps = 1e-4 and the exact log-sum-exp below, as in the NumPy version.  In float32 (GPU
# default) the matrix products lose too much range (exp underflows at e^-87), so the exact
# log-sum-exp is used at every eps, with the matrix-product kernel's 1e-300 floor where
# the float64 code would use it, so that both give the same barycenter.
def make_kernel(N, eps, device, dtype, exact_below=1e-4):
    import torch
    fast = eps >= exact_below * 0.999                  # where the NumPy code used Kernel
    return otlib.LogGibbs(N, eps, device, dtype, exact=not (fast and dtype == torch.float64),
                          floor=fast)


def barycenter(fields, lam, eps, debias=True, eps_start=1e-2, shrink=0.6,
               tol=1e-4, max_iter=20000, stage_iter=300, exact_below=1e-4, verbose=False,
               patience=1000, stage_patience=None, device="auto", dtype="auto",
               return_iterations=False):
    """(Debiased) Sinkhorn barycenter of 2D histograms with eps-scaling.

    Iterations (log domain, as in notebooks/barycenters.ipynb):
        u_k = a_k / (K v_k)
        b   = c * prod_k (K^T u_k)^lam_k        (c = 1 for the classical barycenter)
        v_k = b / (K^T u_k)
        c   = sqrt(c * b / (K c))               (debiasing, Janati et al. 2020)
    Stops when the marginal error sum_k |u_k * (K v_k) - a_k|_1 < tol (as in the notebook),
    or when it has not improved by 2% in `patience` iterations of the last eps
    (`stage_patience` for the earlier ones; None = never).
    Runs on `device` ('auto', 'cuda', 'cpu'); the fields are updated as one batch.
    Returns b (NumPy, float64) and, per input, log u_k and log(K^T u_k) (torch tensors
    of shape (n, N, N), to transport colours) and the kernel K.
    """
    dev = otlib.pick_device(device)                       # (explains a missing PyTorch)
    dt = otlib.pick_dtype(dev, dtype)
    import time
    import torch
    # GPU with dtype auto: each eps stage uses the faster kernel, float64 matrix products
    # while eps >= exact_below (same numbers as the CPU run) and float32 exact log-sum-exp
    # below (the matrix products underflow there).  An explicit --dtype is used throughout.
    per_stage = dev.type == "cuda" and (dtype is None or str(dtype).lower() == "auto")
    stage_dtype = lambda e: (torch.float64 if per_stage and e >= exact_below * 0.999 else dt)
    lam = np.asarray(lam, float) / np.sum(lam)
    # log of the inputs, exactly -inf where there is no mass (a tiny constant instead
    # makes u = a / Kv explode outside the shapes and breaks the stabilisation)
    logA64 = torch.as_tensor(np.stack([np.where(a > 0, np.log(np.where(a > 0, a, 1.0)), -np.inf)
                                       for a in fields]), dtype=torch.float64, device=dev)
    a64 = torch.exp(logA64)                               # the marginals, for the error
    lam64 = torch.as_tensor(lam, dtype=torch.float64, device=dev).view(-1, 1, 1)
    N = fields[0].shape[0]
    lnv = torch.zeros_like(logA64)
    lnc = torch.zeros((N, N), dtype=torch.float64, device=dev)
    schedule = []
    e = max(eps_start, eps)
    while e > eps:
        schedule.append(e); e *= shrink
    schedule.append(eps)
    total, t0 = 0, time.time()
    for s, e in enumerate(schedule):
        if s > 0:                                        # keep the dual potentials eps*log(.)
            r = schedule[s - 1] / e
            lnv = lnv * r; lnc = lnc * r
        sdt = stage_dtype(e)
        logA, lam_t = logA64.to(sdt), lam64.to(sdt)
        lnv, lnc = lnv.to(sdt), lnc.to(sdt)
        K = make_kernel(N, e, dev, sdt, exact_below)
        last = s == len(schedule) - 1
        limit = patience if last else stage_patience
        err, best, best_it = np.inf, np.inf, 0
        for it in range(max_iter if last else stage_iter):
            lnu = logA - K(lnv)
            lnKtu = K(lnu)
            lnb = (lam_t * lnKtu).sum(0)
            if debias:
                lnb = lnb + lnc
                for _ in range(3):
                    lnc = 0.5 * (lnc + lnb - K(lnc))
            else:
                lnb = lnb - torch.logsumexp(lnb.reshape(-1), 0)
            lnv = lnb - lnKtu
            total += 1
            if it % 20 == 19:                            # (summed in float64)
                err = (torch.exp((lnu + K(lnv)).double()) - a64).abs().sum().item()
                if verbose and last and it % 500 == 499:
                    print(f"    eps={e:.2e}  iteration {total}  marginal error {err:.1e}"
                          f"  ({time.time() - t0:.0f}s)", flush=True)
                if err < (tol if last else 1e-3):
                    break
                if err < 0.98 * best:
                    best, best_it = err, it
                elif limit is not None and it - best_it > limit:   # numerical floor reached
                    if verbose and last:
                        print(f"    stopped: marginal error stalled at {err:.1e}", flush=True)
                    break
        if verbose:
            kind = "exact" if K.exact else "matmul"
            print(f"    eps={e:.2e}  iterations so far {total}  marginal error {err:.1e}"
                  f"  [{str(sdt).replace('torch.', '')} {kind}, {time.time() - t0:.0f}s]", flush=True)
    lnb = lnb.double().cpu().numpy()                     # normalise in float64, on the host
    b = np.exp(lnb - logsumexp(lnb))
    return (b, lnu, lnKtu, K, total) if return_iterations else (b, lnu, lnKtu, K)


def transported_colour(colours, lam, lnu, lnKtu, K):
    """Mean colour arriving at each barycenter pixel: [K^T(u_k c_k)] / [K^T u_k], mixed by lam.
    The three channels go through the kernel as one batch."""
    import torch
    out = 0
    for l, c, u, ktu in zip(lam, colours, lnu, lnKtu):
        lc = np.stack([np.log(c[..., ch] + 1e-4) for ch in range(3)])      # (3, N, N)
        lc = torch.as_tensor(lc, dtype=u.dtype, device=u.device)
        logc = (K(u + lc) - ktu).double().cpu().numpy()
        out = out + l * np.exp(np.moveaxis(logc, 0, -1))
    return np.clip(out, 0, 1)


# =============================================================================
# Frames, rendering
# =============================================================================
def frame_path(args, k):
    return os.path.join(args.out, "frames", f"t{k:02d}of{args.steps}_{args.size}_{args.eps:g}"
                        f"{'_debiased' if args.debias else ''}.npz")


def frame(job):
    """Frame k (t = k/steps) from the two aligned inputs [(mass, colour), (mass, colour)]."""
    k, args, ((mc, cc), (mo, co)) = job
    t0 = time.time()
    t = k / args.steps
    if k == 0:
        out = dict(mass=mc, colour=cc)
    elif k == args.steps:
        out = dict(mass=mo, colour=co)
    else:
        b, lnu, lnKtu, K = barycenter([mc, mo], [1 - t, t], args.eps, debias=args.debias,
                                      verbose=args.verbose, device=args.device, dtype=args.dtype)
        out = dict(mass=b, colour=transported_colour([cc, co], [1 - t, t], lnu, lnKtu, K))
    np.savez(frame_path(args, k), **out)
    print(f"  t = {k}/{args.steps} done in {time.time() - t0:.0f}s", flush=True)
    return k, out


def render(fr, q=0.97, px=None):
    m = fr["mass"]
    alpha = np.clip(m / np.quantile(m[m > 0], q), 0, 1)[..., None]
    im = Image.fromarray((fr["colour"] * alpha * 255).astype(np.uint8))
    return im.resize((px, px), Image.LANCZOS) if px else im


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", type=int, default=256, help="grid size (pixels per side)")
    p.add_argument("--eps", type=float, default=1e-4,
                   help="entropic regularization (final). Below 1e-4 the exact (slower) kernel is "
                        "used (on the GPU in float32 it is used at every eps)")
    p.add_argument("--no-debias", dest="debias", action="store_false", help="classical barycenter")
    p.add_argument("--steps", type=int, default=4, help="frames at t = k/steps (default 4)")
    p.add_argument("--workers", type=int, default=os.cpu_count(),
                   help="parallel processes (CPU only; on the GPU frames run one after another)")
    p.add_argument("--out", default=os.path.join(HERE, "out", "meowl"))
    p.add_argument("--verbose", action="store_true")
    otlib.add_device_args(p)
    args = p.parse_args()
    if args.eps < 2.0 / args.size ** 2:
        print(f"warning: eps={args.eps:g} is below one pixel for size {args.size}; "
              f"use eps >= {2.0 / args.size ** 2:.1e} or a larger --size")
    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
    t0 = time.time()
    ks = sorted(range(args.steps + 1), key=lambda k: abs(k - args.steps / 2))   # meowl first
    frames = {k: dict(np.load(frame_path(args, k))) for k in ks if os.path.exists(frame_path(args, k))}
    todo = [k for k in ks if k not in frames]
    if todo:
        dev = otlib.pick_device(args.device)
        args.device = dev.type
        print(f"device: {otlib.describe(dev, otlib.pick_dtype(dev, args.dtype))}", flush=True)
        inputs = aligned("cat", args.size), aligned("owl", args.size)    # segmented once
        jobs = [(k, args, inputs) for k in todo]
        if dev.type == "cuda" or min(args.workers, len(jobs)) <= 1:
            frames.update(frame(j) for j in jobs)         # one GPU: frames one after another
        else:
            from multiprocessing import Pool
            with Pool(min(args.workers, len(jobs))) as pool:
                frames.update(pool.imap_unordered(frame, jobs))
    frames = [frames[k] for k in range(args.steps + 1)]

    px = 768
    if args.steps % 2 == 0:
        render(frames[args.steps // 2], px=px).save(os.path.join(args.out, "meowl.png"))
    tiles = [render(f, px=320) for f in frames]
    strip = Image.new("RGB", (320 * len(tiles), 320))
    for i, im in enumerate(tiles):
        strip.paste(im, (320 * i, 0))
    strip.save(os.path.join(args.out, "strip.png"))
    gif = [render(f, px=480) for f in frames]
    order = list(range(len(gif))) + list(range(len(gif) - 2, 0, -1))
    hold = {0: 900, args.steps // 2: 900, args.steps: 900}
    gif[0].save(os.path.join(args.out, "meowl.gif"), save_all=True,
                append_images=[gif[i] for i in order[1:]],
                duration=[hold.get(i, 130) for i in order], loop=0)
    print(f"done in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
