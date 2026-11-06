"""
meowl_gpu.py -- the sharp cat + owl = meowl (or any --pair of cut-outs), computed on a
CUDA GPU with PyTorch (it used OpenCL before; the CPU is the fallback).

Same maths as photo_barycenter.py (debiased Sinkhorn barycenter, eps-scaling, exact
log-domain kernel), and the same code: the iterations are photo_barycenter.barycenter.
What this script adds: ready-made cut-outs, more photos and alignments (PHOTOS, --pair),
--bg, and the settings below.

Settings that differ from photo_barycenter.py (kept from the OpenCL version)
---------------------------------------------------------------------------
* Kernel.  K = K1 (x) K1 is applied as two log-sum-exp passes
      out[i, j] = log sum_k exp(-(t_i - t_k)^2 / eps + V[k, j])
  computed exactly for every eps (the matrix-product kernel is never used).
* float32 on the GPU.  The log-domain iterations are stable in float32; the
  marginal error bottoms out around 1e-4..1e-3 at small eps (invisible in the
  picture), so the stall test stops as soon as no more progress is being made
  (300 iterations without progress on the last eps, 100 on the others).
* The photos are segmented once per run, not once per frame.

Install
-------
    pip install -r requirements.txt     (PyTorch with CUDA: see README.md)

Usage
-----
    python meowl_gpu.py --list-devices          # check that PyTorch sees the GPU
    python meowl_gpu.py                         # 256 px, eps 1e-4, debiased, 5 frames
    python meowl_gpu.py --size 384 --eps 4e-5   # sharper
    python meowl_gpu.py --size 512 --eps 2e-5 --steps 8 --verbose
    python meowl_gpu.py --no-debias             # classical Sinkhorn barycenter
    python meowl_gpu.py --device cpu            # force the CPU (default: CUDA GPU if any)
    python meowl_gpu.py --pair brown_owl snowy_owl   # other photos (see PHOTOS)

Rule of thumb: keep eps >= 2/size^2 (kernel at least ~1 pixel wide).
Cost per Sinkhorn iteration grows like size^3.

Inputs: the cut-outs listed in PHOTOS, in data/ (cat_cutout.png, owl_cutout.png,
brown_owl_cutout.png, snowy_owl_cutout.png); add an entry there for new ones.
Outputs (folder out/meowl_out/, or out/FROM_TO_out/ for --pair): meowl.png (t = 1/2),
strip.png, meowl.gif, frames/*.npz.  Frames already in the folder are reused.
"""
import argparse
import os
import time

_USER_THREADS = os.environ.get("OMP_NUM_THREADS")   # photo_barycenter sets 1 if unset
import numpy as np
from scipy import ndimage as ndi
from PIL import Image

import otlib
import photo_barycenter as pb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

PHOTOS = {
    # file, GrabCut rectangle (x, y, w, h) around the animal, eye centres (x, y) in pixels,
    # clean-up: "erode" = pixels shaved off the edge (background halo),
    #           "bridge" = remove twigs and repaint across the branch the bird sits on
    # The *_cutout.png files are already cut out (transparent background): their alpha
    # channel is used as the mask and GrabCut is skipped.  A plain photo (no alpha) works
    # too: then the rectangle and clean-up options below are used.
    "cat": ("cat_cutout.png", (55, 30, 450, 580), ((254, 141), (325, 146)), {"erode": 3}),
    "owl": ("owl_cutout.png", (55, 65, 190, 385), ((137, 124), (184, 123)), {"erode": 2}),
    "brown_owl": ("brown_owl_cutout.png", (95, 140, 260, 525), ((134, 205), (199, 203)), {"erode": 2}),
    "snowy_owl": ("snowy_owl_cutout.png", (35, 88, 380, 530), ((253, 157), (321, 153)), {"erode": 2}),
    # portraits: aligned by the eyes (scale, rotation, position) and cut to the same bust
    # (box = half-width and depth below the eyes, in eye distances), see aligned()
    "portrait": ("portrait_cutout.png", (12, 0, 207, 230), ((65, 81), (109, 74)),
                 {"align": "eyes", "box": (2.2, 3.0)}),
    "gigachad": ("gigachad_cutout.png", (5, 15, 382, 377), ((157, 131), (228, 131)),
                 {"align": "eyes", "box": (2.2, 3.0)}),
    # silhouettes (black on white: rendered on a white background).  The two points are
    # landmarks, not eyes: centre of the head and front of the chest.  The horse is fitted
    # to the canvas; the woman is placed so her landmarks land on the horse's ("onto"),
    # which lays her neck along the horse's neck and her hair along its mane.
    "horse": ("horse_cutout.png", (0, 0, 740, 554), ((525, 85), (528, 235)),
              {"align": "fit", "silhouette": True}),
    "woman": ("woman_cutout.png", (0, 0, 236, 315), ((115, 105), (165, 245)),
              {"onto": "horse", "silhouette": True}),
}
FACE_EYE_DIST, FACE_EYE_Y = 0.18, 0.33   # eye distance and eye height on the canvas (fractions)
FIT = 0.9                                # "fit": bounding box fills this much of the canvas
WORK = 1024             # resolution of the aligned canvas before downsampling


# =============================================================================
# Images -> mass + colour on an N x N grid  (unchanged, but segmented only once)
# =============================================================================
_SEGMENTS = {}


def segment(name):
    if name in _SEGMENTS:
        return _SEGMENTS[name]
    import cv2
    fname, rect, _, opt = PHOTOS[name]
    photo = Image.open(os.path.join(DATA, fname))
    if photo.mode in ("RGBA", "LA", "PA") or "transparency" in photo.info:   # ready-made cut-out
        rgba = np.asarray(photo.convert("RGBA"))
        _SEGMENTS[name] = rgba[..., :3], rgba[..., 3] > 127
        return _SEGMENTS[name]
    img = np.asarray(photo.convert("RGB"))
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.grabCut(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), m, rect, np.zeros((1, 65)),
                np.zeros((1, 65)), 8, cv2.GC_INIT_WITH_RECT)
    raw = (m == 1) | (m == 3)
    if opt.get("bridge"):                               # remove twigs, bridge the branch
        mask = ndi.binary_opening(raw, structure=np.ones((5, 5)))
        mask = ndi.binary_closing(mask, structure=np.ones((45, 3)))
    else:                                               # cut the halo of the background
        mask = ndi.binary_erosion(raw, iterations=opt.get("erode", 0)) if opt.get("erode") else raw
    mask = ndi.binary_fill_holes(mask)                  # (also puts back eyes GrabCut missed)
    lab, n = ndi.label(mask)
    mask = lab == (1 + np.argmax(ndi.sum(mask, lab, range(1, n + 1))))
    added = mask & ~raw
    if opt.get("bridge") and added.any():               # repaint where the branch was
        img = cv2.inpaint(img, (added * 255).astype(np.uint8), 7, cv2.INPAINT_TELEA)
    _SEGMENTS[name] = img, mask
    return img, mask


def similarity(p1, p2, q1, q2):
    """2x3 matrix of the scale + rotation + shift sending p1 -> q1 and p2 -> q2."""
    p1, p2, q1, q2 = (np.asarray(p, float) for p in (p1, p2, q1, q2))
    v, w = p2 - p1, q2 - q1
    a = np.arctan2(w[1], w[0]) - np.arctan2(v[1], v[0])
    R = np.hypot(*w) / np.hypot(*v) * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return np.c_[R, q1 - R @ p1]


def placement(name):
    """2x3 matrix from photo pixels to the WORK x WORK canvas."""
    img, mask = segment(name)
    (x1, y1), (x2, y2) = PHOTOS[name][2]
    opt = PHOTOS[name][3]
    if "onto" in opt:                                   # own landmarks -> the other photo's
        ref = opt["onto"]
        S, B = similarity(*PHOTOS[name][2], *PHOTOS[ref][2]), placement(ref)
        return np.c_[B[:, :2] @ S[:, :2], B[:, :2] @ S[:, 2] + B[:, 2]]
    if opt.get("align") == "eyes":                      # eyes levelled at a fixed distance
        d = FACE_EYE_DIST * WORK
        return similarity((x1, y1), (x2, y2), (WORK / 2 - d / 2, FACE_EYE_Y * WORK),
                          (WORK / 2 + d / 2, FACE_EYE_Y * WORK))
    if opt.get("align") == "fit":                       # bounding box centred, FIT of the canvas
        rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
        s = FIT * WORK / max(rows.max() - rows.min(), cols.max() - cols.min())
        return np.array([[s, 0, WORK / 2 - s * (cols.min() + cols.max()) / 2],
                         [0, s, WORK / 2 - s * (rows.min() + rows.max()) / 2]])
    rows = np.where(mask.any(1))[0]                     # animals: 86% of the height, eyes on top
    s = 0.86 * WORK / (rows.max() - rows.min())
    return np.array([[s, 0, WORK / 2 - s * (x1 + x2) / 2], [0, s, 0.19 * WORK - s * (y1 + y2) / 2]])


def aligned(name, N):
    """Mass and colour of the photo on an N x N grid, placed on the canvas by placement():
    animals scaled to 86% of the canvas height with the eye midpoint at (0.5, 0.19);
    portraits ("align": "eyes") with the eyes levelled, FACE_EYE_DIST apart, then cut to
    the same bust box; "fit" centred; "onto" landmarks matched to another photo's."""
    import cv2
    img, mask = segment(name)
    opt = PHOTOS[name][3]
    A = np.float32(placement(name))
    warp = lambda x: cv2.warpAffine(x, A, (WORK, WORK), flags=cv2.INTER_AREA)
    m = warp(mask.astype(np.float32))
    if "box" in opt:                                    # same bust for both portraits
        half, depth = opt["box"]
        d = FACE_EYE_DIST * WORK
        m[:, np.abs(np.arange(WORK) + 0.5 - WORK / 2) > half * d] = 0
        m[int(FACE_EYE_Y * WORK + depth * d):] = 0
    rgb = warp(img.astype(np.float32) / 255)
    down = lambda x: cv2.resize(x, (N, N), interpolation=cv2.INTER_AREA)
    mass = down(m)
    col = down(rgb * m[..., None]) / np.maximum(mass[..., None], 1e-6)
    return mass / mass.sum(), np.clip(col, 0, 1)


# =============================================================================
# Device (PyTorch: CUDA, or the CPU as fallback)
# =============================================================================
def list_devices():
    import torch
    out = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            pr = torch.cuda.get_device_properties(i)
            out.append(f"cuda:{i}  {pr.name}  ({pr.multi_processor_count} SMs, "
                       f"{pr.total_memory // 2**20} MiB, compute capability {pr.major}.{pr.minor})")
    return out


# =============================================================================
# (Debiased) Sinkhorn barycenter: eps-scaling, warm start (photo_barycenter.barycenter)
# =============================================================================
def barycenter(fields, lam, eps, debias=True, eps_start=1e-2, shrink=0.6,
               tol=1e-4, max_iter=20000, stage_iter=300, patience=300, verbose=False,
               device="auto", dtype="auto"):
    """(Debiased) Sinkhorn barycenter of 2D histograms (photo_barycenter.barycenter with
    this script's settings: exact kernel at every eps, stall test after `patience`
    iterations on the last eps and 100 on the others).
    Returns b (host, float64), log u_k and log(K^T u_k) (on the device, for the colours),
    the kernel and the number of iterations.
    """
    return pb.barycenter(fields, lam, eps, debias=debias, eps_start=eps_start, shrink=shrink,
                         tol=tol, max_iter=max_iter, stage_iter=stage_iter,
                         exact_below=np.inf, patience=patience, stage_patience=100,
                         verbose=verbose, device=device, dtype=dtype, return_iterations=True)


transported_colour = pb.transported_colour     # [K^T(u_k c_k)] / [K^T u_k], mixed by lam


# =============================================================================
# Frames, rendering
# =============================================================================
def frame_path(args, k):
    return os.path.join(args.out, "frames", f"t{k:02d}of{args.steps}_{args.size}_{args.eps:g}"
                        f"{'_debiased' if args.debias else ''}.npz")


def render(fr, q=0.97, px=None, bg=0.0):
    m = fr["mass"]
    alpha = np.clip(m / np.quantile(m[m > 0], q), 0, 1)[..., None]
    im = Image.fromarray(((fr["colour"] * alpha + bg * (1 - alpha)) * 255).astype(np.uint8))
    return im.resize((px, px), Image.LANCZOS) if px else im


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", type=int, default=256, help="grid size (pixels per side)")
    p.add_argument("--eps", type=float, default=1e-4, help="entropic regularization (final)")
    p.add_argument("--no-debias", dest="debias", action="store_false", help="classical barycenter")
    p.add_argument("--steps", type=int, default=4, help="frames at t = k/steps (default 4)")
    p.add_argument("--pair", nargs=2, default=["cat", "owl"], metavar=("FROM", "TO"),
                   choices=sorted(PHOTOS), help="the two photos to interpolate (default: cat owl)")
    p.add_argument("--out", default=None,
                   help="output folder (default: meowl_out for cat owl, FROM_TO_out otherwise)")
    otlib.add_device_args(p)
    p.add_argument("--list-devices", action="store_true", help="list CUDA devices and exit")
    p.add_argument("--bg", choices=["black", "white"], default=None,
                   help="background of the pictures (default: white for silhouettes, else black)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.out is None:                         # frames are cached per folder: one per pair
        name = "meowl" if args.pair == ["cat", "owl"] else "_".join(args.pair)
        args.out = os.path.join(HERE, "out", f"{name}_out")
    if args.bg is None:
        args.bg = "white" if all(PHOTOS[n][3].get("silhouette") for n in args.pair) else "black"
    bg = 1.0 if args.bg == "white" else 0.0

    if args.list_devices:
        import torch
        devs = list_devices()
        print(f"PyTorch {torch.__version__}, built with CUDA {torch.version.cuda}")
        if not devs:
            otlib.pick_device("cuda")                # prints why there is no GPU
        for d in devs:
            print(d)
        return
    if args.eps < 2.0 / args.size ** 2:
        print(f"warning: eps={args.eps:g} is below one pixel for size {args.size}; "
              f"use eps >= {2.0 / args.size ** 2:.1e} or a larger --size")
    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
    t0 = time.time()

    (mc, cc), (mo, co) = aligned(args.pair[0], args.size), aligned(args.pair[1], args.size)
    frames = {0: dict(mass=mc, colour=cc), args.steps: dict(mass=mo, colour=co)}
    todo = []
    for k in sorted(range(1, args.steps), key=lambda k: abs(k - args.steps / 2)):   # meowl first
        if os.path.exists(frame_path(args, k)):
            d = np.load(frame_path(args, k))
            frames[k] = dict(mass=d["mass"], colour=d["colour"])
        else:
            todo.append(k)

    if todo:
        dev = otlib.pick_device(args.device)
        if dev.type == "cpu" and _USER_THREADS is None:   # CPU fallback: use every core
            import torch
            torch.set_num_threads(os.cpu_count())
        print(f"device: {otlib.describe(dev, otlib.pick_dtype(dev, args.dtype))}", flush=True)
        for k in todo:                               # one GPU: frames one after another
            t1 = time.time()
            t = k / args.steps
            b, lnu, lnKtu, K, its = barycenter([mc, mo], [1 - t, t], args.eps, debias=args.debias,
                                               verbose=args.verbose, device=dev, dtype=args.dtype)
            colour = transported_colour([cc, co], [1 - t, t], lnu, lnKtu, K)
            frames[k] = dict(mass=b, colour=colour)
            np.savez(frame_path(args, k), mass=b, colour=colour)
            print(f"  t = {k}/{args.steps} done in {time.time() - t1:.1f}s ({its} iterations)",
                  flush=True)
    frames = [frames[k] for k in range(args.steps + 1)]

    px = 768
    if args.steps % 2 == 0:
        render(frames[args.steps // 2], px=px, bg=bg).save(os.path.join(args.out, "meowl.png"))
    tiles = [render(f, px=320, bg=bg) for f in frames]
    strip = Image.new("RGB", (320 * len(tiles), 320))
    for i, im in enumerate(tiles):
        strip.paste(im, (320 * i, 0))
    strip.save(os.path.join(args.out, "strip.png"))
    gif = [render(f, px=480, bg=bg) for f in frames]
    order = list(range(len(gif))) + list(range(len(gif) - 2, 0, -1))
    hold = {0: 900, args.steps // 2: 900, args.steps: 900}
    gif[0].save(os.path.join(args.out, "meowl.gif"), save_all=True,
                append_images=[gif[i] for i in order[1:]],
                duration=[hold.get(i, 130) for i in order], loop=0)
    print(f"done in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
