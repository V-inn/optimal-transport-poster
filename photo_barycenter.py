"""
photo_barycenter.py -- Wasserstein barycenter of two photos (cat + owl = "meowl"), on the CPU.

Also the library used by barycenter_triangle.py (segmentation, alignment, debiased Sinkhorn
barycenter, colours carried along the transport plan).  meowl_gpu.py is the same algorithm
on the GPU (OpenCL).

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
    pip install numpy scipy pillow opencv-python
    python photo_barycenter.py                       # defaults: 256 px, eps 1e-4, debiased
    python photo_barycenter.py --size 384 --eps 4e-5 # sharper, much slower (exact kernel)
    python photo_barycenter.py --steps 8             # 9 frames (t = k/8) for a smoother GIF
    python photo_barycenter.py --no-debias           # classical Sinkhorn barycenter

Rule of thumb: keep eps >= 2/size^2 (kernel at least ~1 pixel wide).
Down to eps = 1e-4 a fast matrix-product kernel is used (a few minutes per frame on one
core); below that it switches to an exact log-sum-exp kernel, which is stable but costs
size^3 per step (expect tens of minutes per frame at 256 px).  Frames run in parallel.

Inputs: data/cat_photo.png and data/owl_photo.png (edit PHOTOS for others).
Outputs (folder out/meowl/): meowl.png (t = 1/2), strip.png, meowl.gif, frames/*.npz
"""
import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")       # frames run in parallel processes
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
from scipy import ndimage as ndi
from scipy.special import logsumexp
from PIL import Image

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
class Kernel:
    """K = K1 (x) K1 is separable, so K*V costs two N x N matrix products.
    Each product is stabilised by the max of the log-values along the summed axis."""
    def __init__(self, N, eps):
        t = np.linspace(0.0, 1.0, N)
        self.K1 = np.exp(-((t[:, None] - t[None, :]) ** 2) / eps)

    def __call__(self, lnV):                             # returns log(K exp(lnV))
        m1 = lnV.max(axis=0, keepdims=True)
        m1 = np.where(np.isfinite(m1), m1, 0.0)         # columns with no mass at all
        A = np.log(np.maximum(self.K1 @ np.exp(lnV - m1), 1e-300)) + m1
        m2 = A.max(axis=1, keepdims=True)
        return np.log(np.maximum(np.exp(A - m2) @ self.K1.T, 1e-300)) + m2


class ExactKernel:
    """Same operator with an exact log-sum-exp (N^3 work per call, but stable for any eps).
    The fast Kernel loses accuracy below eps ~ 1e-4, where the scalings span > 700 in log."""
    def __init__(self, N, eps, chunk=32):
        t = np.linspace(0.0, 1.0, N)
        self.M = -((t[:, None] - t[None, :]) ** 2) / eps
        self.chunk = chunk

    def _lse(self, lnV):                                 # out[i, j] = LSE_k M[i, k] + lnV[k, j]
        out = np.empty_like(lnV)
        for s in range(0, lnV.shape[0], self.chunk):
            out[s:s + self.chunk] = logsumexp(self.M[s:s + self.chunk, :, None] + lnV[None], axis=1)
        return out

    def __call__(self, lnV):
        return self._lse(self._lse(lnV).T).T


def barycenter(fields, lam, eps, debias=True, eps_start=1e-2, shrink=0.6,
               tol=1e-4, max_iter=20000, stage_iter=300, exact_below=1e-4, verbose=False):
    """(Debiased) Sinkhorn barycenter of 2D histograms with eps-scaling.

    Iterations (log domain, as in notebooks/barycenters.ipynb):
        u_k = a_k / (K v_k)
        b   = c * prod_k (K^T u_k)^lam_k        (c = 1 for the classical barycenter)
        v_k = b / (K^T u_k)
        c   = sqrt(c * b / (K c))               (debiasing, Janati et al. 2020)
    Stops when the marginal error sum_k |u_k * (K v_k) - a_k|_1 < tol (as in the notebook).
    Returns b and, per input, log u_k and log(K^T u_k) (to transport colours).
    """
    lam = np.asarray(lam, float) / np.sum(lam)
    # log of the inputs, exactly -inf where there is no mass (a tiny constant instead
    # makes u = a / Kv explode outside the shapes and breaks the stabilisation)
    logA = [np.where(a > 0, np.log(np.where(a > 0, a, 1.0)), -np.inf) for a in fields]
    N = fields[0].shape[0]
    lnv = [np.zeros((N, N)) for _ in fields]
    lnc = np.zeros((N, N))
    schedule = []
    e = max(eps_start, eps)
    while e > eps:
        schedule.append(e); e *= shrink
    schedule.append(eps)
    total = 0
    for s, e in enumerate(schedule):
        if s > 0:                                        # keep the dual potentials eps*log(.)
            r = schedule[s - 1] / e
            lnv = [v * r for v in lnv]; lnc = lnc * r
        K = (ExactKernel if e < exact_below * 0.999 else Kernel)(N, e)
        last = s == len(schedule) - 1
        err, best, best_it = np.inf, np.inf, 0
        for it in range(max_iter if last else stage_iter):
            lnu = [la - K(v) for la, v in zip(logA, lnv)]
            lnKtu = [K(u) for u in lnu]
            lnb = sum(l * x for l, x in zip(lam, lnKtu))
            if debias:
                lnb = lnb + lnc
                for _ in range(3):
                    lnc = 0.5 * (lnc + lnb - K(lnc))
            else:
                lnb = lnb - logsumexp(lnb)
            lnv = [lnb - x for x in lnKtu]
            total += 1
            if it % 20 == 19:
                err = sum(np.abs(np.exp(u + K(v)) - np.exp(la)).sum()
                          for u, v, la in zip(lnu, lnv, logA))
                if verbose and last and it % 500 == 499:
                    print(f"    eps={e:.2e}  iteration {total}  marginal error {err:.1e}", flush=True)
                if err < (tol if last else 1e-3):
                    break
                if err < 0.98 * best:
                    best, best_it = err, it
                elif last and it - best_it > 1000:          # no progress: numerical floor reached
                    if verbose:
                        print(f"    stopped: marginal error stalled at {err:.1e}", flush=True)
                    break
        if verbose:
            print(f"    eps={e:.2e}  iterations so far {total}  marginal error {err:.1e}", flush=True)
    b = np.exp(lnb - logsumexp(lnb))
    return b, lnu, lnKtu, K


def transported_colour(colours, lam, lnu, lnKtu, K):
    """Mean colour arriving at each barycenter pixel: [K^T(u_k c_k)] / [K^T u_k], mixed by lam."""
    out = 0
    for l, c, u, ktu in zip(lam, colours, lnu, lnKtu):
        out = out + l * np.stack([np.exp(K(u + np.log(c[..., ch] + 1e-4)) - ktu) for ch in range(3)], -1)
    return np.clip(out, 0, 1)


# =============================================================================
# Frames, rendering
# =============================================================================
def frame(job):
    k, args = job
    path = os.path.join(args.out, "frames", f"t{k:02d}of{args.steps}_{args.size}_{args.eps:g}"
                        f"{'_debiased' if args.debias else ''}.npz")
    if os.path.exists(path):
        return k, dict(np.load(path))
    t0 = time.time()
    (mc, cc), (mo, co) = aligned("cat", args.size), aligned("owl", args.size)
    t = k / args.steps
    if k == 0:
        out = dict(mass=mc, colour=cc)
    elif k == args.steps:
        out = dict(mass=mo, colour=co)
    else:
        b, lnu, lnKtu, K = barycenter([mc, mo], [1 - t, t], args.eps, debias=args.debias,
                                      verbose=args.verbose)
        out = dict(mass=b, colour=transported_colour([cc, co], [1 - t, t], lnu, lnKtu, K))
    np.savez(path, **out)
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
                   help="entropic regularization (final). Below 1e-4 the exact (slower) kernel is used")
    p.add_argument("--no-debias", dest="debias", action="store_false", help="classical barycenter")
    p.add_argument("--steps", type=int, default=4, help="frames at t = k/steps (default 4)")
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel processes")
    p.add_argument("--out", default=os.path.join(HERE, "out", "meowl"))
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.eps < 2.0 / args.size ** 2:
        print(f"warning: eps={args.eps:g} is below one pixel for size {args.size}; "
              f"use eps >= {2.0 / args.size ** 2:.1e} or a larger --size")
    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
    t0 = time.time()
    ks = sorted(range(args.steps + 1), key=lambda k: abs(k - args.steps / 2))   # meowl first
    from multiprocessing import Pool
    with Pool(max(1, min(args.workers, len(ks)))) as pool:
        frames = dict(pool.imap_unordered(frame, [(k, args) for k in ks]))
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
