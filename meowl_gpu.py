"""
meowl_gpu.py -- the sharp cat + owl = meowl, computed on the GPU (OpenCL).

Same maths as photo_barycenter.py (debiased Sinkhorn barycenter, eps-scaling, exact
log-domain kernel), but the iterations run on the GPU through OpenCL, which is
the most reliable way to drive an Intel Iris Xe from Python.

What changed compared with photo_barycenter.py
-----------------------------------------
* GPU kernel.  K = K1 (x) K1 is applied as two fused log-sum-exp passes
      out[i, j] = log sum_k exp(-(t_i - t_k)^2 / eps + V[k, j])
  computed on the fly (no N x N kernel matrix, no N^3 temporaries), tiled in
  local memory with a blocked online max.  It is exact for every eps, so the
  fast-but-fragile matrix-product kernel is gone.
* float32.  Iris Xe has no double precision in hardware.  The log-domain
  iterations are stable in float32; the marginal error bottoms out around
  1e-4..1e-3 at small eps (invisible in the picture), so the stall test
  stops as soon as no more progress is being made.
* The photos are segmented once per run, not once per frame.

Install
-------
    pip install numpy scipy pillow opencv-python pyopencl

  Windows: the OpenCL driver for Iris Xe ships with the Intel graphics driver.
  Linux (Ubuntu/Debian): sudo apt install intel-opencl-icd   (then `clinfo` should list it)

Usage
-----
    python meowl_gpu.py --list-devices          # check that the Iris Xe is visible
    python meowl_gpu.py                         # 256 px, eps 1e-4, debiased, 5 frames
    python meowl_gpu.py --size 384 --eps 4e-5   # sharper
    python meowl_gpu.py --size 512 --eps 2e-5 --steps 8 --verbose
    python meowl_gpu.py --no-debias             # classical Sinkhorn barycenter
    python meowl_gpu.py --device cpu            # any OpenCL device whose name matches
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

import numpy as np
from scipy import ndimage as ndi
from scipy.special import logsumexp
from PIL import Image

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
NEG = -1e30             # "log 0": finite, so no inf/nan can appear on the GPU


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
# OpenCL: fused log-sum-exp kernels
# =============================================================================
KERNEL_SRC = r"""
#ifdef EXACT_EXP
#define EXPF(x) exp(x)
#else
#define EXPF(x) native_exp(x)
#endif
#define NEG (-1e30f)

/* Adds the T terms  -c (pos - k0 - q)^2 + t[q * stride]  to a running log-sum-exp
   (m, s): max of the tile first, rescale the running sum once, then one exp per term. */
inline void lse_tile(__local const float *t, const int stride, const int pos, const int k0,
                     const float c, float *m, float *s)
{
    float x[T], cm = NEG;
    #pragma unroll
    for (int q = 0; q < T; q++) {
        const float d = (float)(pos - k0 - q);
        x[q] = t[q * stride] - c * d * d;
        cm = fmax(cm, x[q]);
    }
    const float nm = fmax(*m, cm);
    float acc = *s * EXPF(*m - nm);
    #pragma unroll
    for (int q = 0; q < T; q++) acc += EXPF(x[q] - nm);
    *s = acc; *m = nm;
}

/* out[i, j] = log sum_k exp(-c (i - k)^2 + V[k, j])     (sum over the row index)
   One work-item per output; k runs over tiles of T rows staged in local memory. */
__kernel __attribute__((reqd_work_group_size(T, T, 1)))
void lse_rows(__global const float *V, __global float *out, const int N, const float c)
{
    const int lj = get_local_id(0), li = get_local_id(1);
    const int j = get_group_id(0) * T + lj, i = get_group_id(1) * T + li;
    __local float tile[T][T];
    float m = NEG, s = 0.0f;
    for (int k0 = 0; k0 < N; k0 += T) {
        const int k = k0 + li;
        tile[li][lj] = (k < N && j < N) ? V[k * N + j] : NEG;
        barrier(CLK_LOCAL_MEM_FENCE);
        lse_tile(&tile[0][lj], T, i, k0, c, &m, &s);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (i < N && j < N) out[i * N + j] = m + log(s);
}

/* out[i, j] = log sum_k exp(-c (j - k)^2 + V[i, k])     (sum over the column index) */
__kernel __attribute__((reqd_work_group_size(T, T, 1)))
void lse_cols(__global const float *V, __global float *out, const int N, const float c)
{
    const int lj = get_local_id(0), li = get_local_id(1);
    const int j = get_group_id(0) * T + lj, i = get_group_id(1) * T + li;
    __local float tile[T][T + 1];
    float m = NEG, s = 0.0f;
    for (int k0 = 0; k0 < N; k0 += T) {
        const int k = k0 + lj;
        tile[li][lj] = (k < N && i < N) ? V[i * N + k] : NEG;
        barrier(CLK_LOCAL_MEM_FENCE);
        lse_tile(&tile[li][0], 1, j, k0, c, &m, &s);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (i < N && j < N) out[i * N + j] = m + log(s);
}
"""


def list_devices():
    import pyopencl as cl
    try:
        platforms = cl.get_platforms()
    except cl.Error:
        platforms = []
    out = []
    for p in platforms:
        for d in p.get_devices():
            out.append((p, d))
    return out


class GPU:
    """OpenCL context + the operations the Sinkhorn loop needs (float32, device-resident)."""

    def __init__(self, hint=None, exact_exp=False):
        import pyopencl as cl
        import pyopencl.array as cla
        from pyopencl.tools import MemoryPool, ImmediateAllocator
        self.cl, self.cla = cl, cla
        devs = list_devices()
        if not devs:
            raise SystemExit(
                "No OpenCL device found.\n"
                "  Windows: update the Intel graphics driver (it contains the OpenCL runtime).\n"
                "  Linux:   sudo apt install intel-opencl-icd   (check with: clinfo)")
        if hint:
            h = hint.lower()
            pick = [(p, d) for p, d in devs if
                    (h == "gpu" and d.type & cl.device_type.GPU) or
                    (h == "cpu" and d.type & cl.device_type.CPU) or
                    h in d.name.lower() or h in p.name.lower()]
            if not pick:
                raise SystemExit(f"no OpenCL device matches '{hint}'; try --list-devices")
        else:
            pick = [(p, d) for p, d in devs if d.type & cl.device_type.GPU] or devs
            if not pick[0][1].type & cl.device_type.GPU:
                print("warning: no OpenCL GPU found, running on", pick[0][1].name.strip())
        self.device = pick[0][1]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)
        self.alloc = MemoryPool(ImmediateAllocator(self.queue))
        self.T = 16 if self.device.max_work_group_size >= 256 else 8
        opts = [f"-DT={self.T}"] + (["-DEXACT_EXP"] if exact_exp else [])
        self.prg = cl.Program(self.ctx, KERNEL_SRC).build(options=opts)
        self._rows, self._cols = self.prg.lse_rows, self.prg.lse_cols

    @property
    def name(self):
        return self.device.name.strip()

    def put(self, x):
        return self.cla.to_device(self.queue, np.ascontiguousarray(x, np.float32), allocator=self.alloc)

    def K(self, V, eps):
        """log(K exp(V)) for the Gibbs kernel exp(-|x-y|^2/eps) on linspace(0,1,N)^2."""
        N = V.shape[0]
        c = np.float32(1.0 / ((N - 1) ** 2 * eps))       # (t_i - t_k)^2/eps = c (i - k)^2
        g = ((N + self.T - 1) // self.T) * self.T
        tmp = self.cla.empty_like(V)
        out = self.cla.empty_like(V)
        self._rows(self.queue, (g, g), (self.T, self.T), V.data, tmp.data, np.int32(N), c)
        self._cols(self.queue, (g, g), (self.T, self.T), tmp.data, out.data, np.int32(N), c)
        return out

    # The two reductions are done on the host (N^2 numbers, cheap on an integrated GPU,
    # and in float64): |u * (K v) - a|_1 every 20 iterations, and the global log-sum-exp
    # that normalises b (classical barycenter only).
    @staticmethod
    def marginal_error(u, Kv, a):
        return float(np.abs(np.exp(u.get().astype(np.float64) + Kv.get()) - a).sum())

    @staticmethod
    def logsumexp(x):
        return float(logsumexp(x.get().astype(np.float64)))


# =============================================================================
# (Debiased) Sinkhorn barycenter: eps-scaling, coarse-to-fine, warm start
# =============================================================================
def barycenter(gpu, fields, lam, eps, debias=True, eps_start=1e-2, shrink=0.6,
               tol=1e-4, max_iter=20000, stage_iter=300, patience=300, verbose=False):
    """(Debiased) Sinkhorn barycenter of 2D histograms, iterated on the GPU.

    Iterations (log domain, as in notebooks/barycenters.ipynb):
        u_k = a_k / (K v_k)
        b   = c * prod_k (K^T u_k)^lam_k        (c = 1 for the classical barycenter)
        v_k = b / (K^T u_k)
        c   = sqrt(c * b / (K c))               (debiasing, Janati et al. 2020)
    Stops when the marginal error sum_k |u_k * (K v_k) - a_k|_1 < tol, or when it has
    not improved for `patience` iterations (float32 floor).
    Returns b (host, float64), log u_k and log(K^T u_k) (on the GPU, for the colours)
    and the number of iterations.
    """
    lam = np.asarray(lam, float) / np.sum(lam)
    N = fields[0].shape[0]
    schedule, e = [], max(eps_start, eps)
    while e > eps:
        schedule.append(e); e *= shrink
    schedule.append(eps)
    # log of the inputs, NEG (= log 0) where there is no mass (a tiny constant instead
    # makes u = a / Kv explode outside the shapes and breaks the stabilisation)
    logA = [gpu.put(np.where(a > 0, np.log(np.where(a > 0, a, 1.0)), NEG)) for a in fields]
    lnv = [gpu.put(np.zeros((N, N))) for _ in fields]
    lnc = gpu.put(np.zeros((N, N)))
    lam32 = [np.float32(l) for l in lam]
    total = 0
    for s, e in enumerate(schedule):
        if s > 0:                                    # keep the dual potentials eps*log(.)
            r = np.float32(schedule[s - 1] / e)
            lnv = [v * r for v in lnv]; lnc = lnc * r
        last = s == len(schedule) - 1
        err, best, best_it = np.inf, np.inf, 0
        for it in range(max_iter if last else stage_iter):
            lnu = [la - gpu.K(v, e) for la, v in zip(logA, lnv)]
            lnKtu = [gpu.K(u, e) for u in lnu]
            lnb = lnKtu[0] * lam32[0]
            for l, x in zip(lam32[1:], lnKtu[1:]):
                lnb = lnb + x * l
            if debias:
                lnb = lnb + lnc
                for _ in range(3):
                    lnc = (lnc + lnb - gpu.K(lnc, e)) * np.float32(0.5)
            else:
                lnb = lnb - np.float32(gpu.logsumexp(lnb))
            lnv = [lnb - x for x in lnKtu]
            total += 1
            if it % 20 == 19:
                err = sum(gpu.marginal_error(u, gpu.K(v, e), a) for u, v, a in zip(lnu, lnv, fields))
                if verbose and last and it % 500 == 499:
                    print(f"    eps={e:.2e}  iteration {total}  marginal error {err:.1e}", flush=True)
                if err < (tol if last else 1e-3):
                    break
                if err < 0.98 * best:
                    best, best_it = err, it
                elif it - best_it > (patience if last else 100):   # float32 floor reached
                    break
        if verbose:
            print(f"    eps={e:.2e}  iterations so far {total}  marginal error {err:.1e}", flush=True)
    lnb_h = lnb.get().astype(np.float64)
    b = np.exp(lnb_h - logsumexp(lnb_h))
    return b, lnu, lnKtu, total


def transported_colour(gpu, colours, lam, lnu, lnKtu, eps):
    """Mean colour arriving at each barycenter pixel: [K^T(u_k c_k)] / [K^T u_k], mixed by lam."""
    out = 0
    for l, c, u, ktu in zip(lam, colours, lnu, lnKtu):
        chans = []
        for ch in range(3):
            lc = gpu.put(np.log(c[..., ch] + 1e-4))
            chans.append(np.exp((gpu.K(u + lc, eps) - ktu).get().astype(np.float64)))
        out = out + l * np.stack(chans, -1)
    return np.clip(out, 0, 1)


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
    p.add_argument("--device", default=None,
                   help="OpenCL device to use: 'gpu', 'cpu' or part of its name (default: first GPU)")
    p.add_argument("--list-devices", action="store_true", help="list OpenCL devices and exit")
    p.add_argument("--exact-exp", action="store_true",
                   help="use the full-precision exp in the kernel (slower; only if results look off)")
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
        devs = list_devices()
        if not devs:
            print("no OpenCL devices found")
        for i, (pl, d) in enumerate(devs):
            print(f"[{i}] {d.name.strip()}  ({pl.name.strip()}, "
                  f"{d.max_compute_units} compute units, {d.global_mem_size // 2**20} MiB)")
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
        gpu = GPU(args.device, args.exact_exp)
        print(f"OpenCL device: {gpu.name}", flush=True)
        for k in todo:                               # one GPU: frames one after another
            t1 = time.time()
            t = k / args.steps
            b, lnu, lnKtu, its = barycenter(gpu, [mc, mo], [1 - t, t], args.eps,
                                            debias=args.debias, verbose=args.verbose)
            colour = transported_colour(gpu, [cc, co], [1 - t, t], lnu, lnKtu, args.eps)
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
