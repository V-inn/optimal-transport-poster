"""
otlib.py -- small toolbox used to generate the poster figures.

It follows the same algorithms as the notebooks in
https://github.com/AI-Quantum-Research-Group/OT-Jan-2026 (intro, sinkhorn,
barycenters), written in NumPy (and PyTorch for the grid kernels, so that they
run on a CUDA GPU when there is one) without JAX / uot-bench:

  * sinkhorn_log          -- log-domain Sinkhorn for entropic OT (dense cost, NumPy)
  * pick_device           -- --device auto|cuda|cpu -> torch device (CPU fallback)
  * LogGibbs              -- log(K exp(V)) for the Gibbs kernel exp(-|x-y|^2/eps) on
                             a d x d grid.  K = K1 (x) K1 is separable, so it is two
                             passes, one along each axis, batched over many fields
  * grid_barycenter_log   -- the log-domain Sinkhorn barycenter of
                             notebooks/barycenters.ipynb, on a d x d image grid,
                             using LogGibbs.  This is what makes 128x128 shapes feasible.
  * load_shape            -- image -> probability measure on a grid
                             (same idea as uot.data.load_image_as_binary_grid)
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from PIL import Image

try:                                    # only the grid kernels need PyTorch
    import torch
except ImportError:                     # make_figures.py / color_transfer.py work without it
    torch = None

TINY = 1e-300
LOG_TINY = float(np.log(TINY))          # -690.78: floor of the matrix-product kernel


# ----------------------------------------------------------------------------
# Entropic OT between two histograms (dense cost matrix)
# ----------------------------------------------------------------------------
def sinkhorn_log(a, b, C, eps, n_iter=2000, tol=1e-9, record=False):
    """Log-domain Sinkhorn.  Returns P, f, g (and the marginal errors if record).

    Solves  min_P <C,P> + eps * sum P (log P - 1)  s.t.  P1 = a, P^T 1 = b,
    whose solution is P_ij = exp((f_i + g_j - C_ij)/eps).
    """
    la, lb = np.log(np.maximum(a, TINY)), np.log(np.maximum(b, TINY))
    f = np.zeros_like(a, dtype=float)
    g = np.zeros_like(b, dtype=float)
    errs = []
    for it in range(n_iter):
        f = eps * la - eps * logsumexp((g[None, :] - C) / eps, axis=1)
        g = eps * lb - eps * logsumexp((f[:, None] - C) / eps, axis=0)
        if record or it % 20 == 0:
            P = np.exp((f[:, None] + g[None, :] - C) / eps)
            err = np.abs(P.sum(1) - a).sum()
            if record:
                errs.append(err)
            if err < tol:
                break
    P = np.exp((f[:, None] + g[None, :] - C) / eps)
    return (P, f, g, np.array(errs)) if record else (P, f, g)


# ----------------------------------------------------------------------------
# Device selection (PyTorch): --device auto | cuda | cpu
# ----------------------------------------------------------------------------
DEVICES = ("auto", "cuda", "cpu")
DTYPES = ("auto", "float32", "float64")


def _need_torch():
    if torch is None:
        raise SystemExit("PyTorch is needed for the barycenters: pip install torch "
                         "(see README.md for the CUDA build)")


def pick_device(choice="auto", verbose=True):
    """'auto': the CUDA GPU if PyTorch sees one, else the CPU.  'cuda' without a usable
    GPU falls back to the CPU too (with a warning), so the scripts always run."""
    _need_torch()
    if isinstance(choice, torch.device):
        return choice
    choice = (choice or "auto").lower()
    if choice not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, not {choice!r}")
    if choice == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if verbose:
        why = ("this PyTorch was built without CUDA (CPU-only wheel, see README.md)"
               if torch.version.cuda is None else
               f"PyTorch {torch.__version__} has CUDA {torch.version.cuda} but sees no GPU "
               "(driver too old or missing? check nvidia-smi)")
        print(f"{'warning' if choice == 'cuda' else 'note'}: no CUDA GPU: {why}; using the CPU",
              flush=True)
    return torch.device("cpu")


def pick_dtype(device, choice="auto"):
    """'auto': float32 on the GPU (consumer/workstation GPUs have slow float64),
    float64 on the CPU (as the original NumPy code)."""
    _need_torch()
    if isinstance(choice, torch.dtype):
        return choice
    choice = (choice or "auto").lower()
    if choice not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, not {choice!r}")
    if choice == "auto":
        return torch.float32 if device.type == "cuda" else torch.float64
    return getattr(torch, choice)


def add_device_args(parser):
    """--device and --dtype, shared by the scripts."""
    parser.add_argument("--device", choices=DEVICES, default="auto",
                        help="where the Sinkhorn iterations run: auto = CUDA GPU if available, "
                             "else CPU (default auto)")
    parser.add_argument("--dtype", choices=DTYPES, default="auto",
                        help="auto = on the GPU float64 matrix products for eps >= 1e-4 and "
                             "float32 exact log-sum-exp below; float64 on the CPU (default auto)")


def describe(device, dtype):
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    return f"{device.type} ({name}), {str(dtype).replace('torch.', '')}"


# ----------------------------------------------------------------------------
# The separable Gibbs kernel on a d x d grid, in the log domain (PyTorch)
# ----------------------------------------------------------------------------
# On CUDA the exact log-sum-exp is compiled with torch.compile: the broadcast sum
# X[..., i, None, j] + M[l, j] is then computed inside the reduction kernel instead of
# being written to memory (d^3 numbers per field), which is what made the eager version
# slow.  Set OTLIB_NO_COMPILE=1 to use the eager version; it is also used automatically
# if compiling fails (e.g. no Triton).
_LSE = {"fn": None, "off": False}


def _lse_eager(X, M):
    return torch.logsumexp(X.unsqueeze(-2) + M, dim=-1)


def _compiled_lse():
    import os
    if _LSE["off"] or os.environ.get("OTLIB_NO_COMPILE"):
        return None
    if _LSE["fn"] is None:
        try:
            _LSE["fn"] = torch.compile(_lse_eager, dynamic=False)
        except Exception as err:
            _disable_compiled_lse(err)
            return None
    return _LSE["fn"]


def _disable_compiled_lse(err):
    if not _LSE["off"]:
        print(f"[otlib] torch.compile not available ({type(err).__name__}: {err}); "
              "using the slower eager kernel", flush=True)
    _LSE["off"] = True

class LogGibbs:
    """V -> log(K exp(V)) for K = K1 (x) K1, K1[i, k] = exp(-(t_i - t_k)^2 / eps),
    t = linspace(0, 1, d).  V has shape (..., d, d): any number of fields at once.
    Because K is separable this is two passes, one log-sum-exp along each axis:
        A[i, j]   = log sum_k K1[i, k] exp(V[k, j])
        out[i, l] = log sum_j exp(A[i, j]) K1[l, j]
    V may contain -inf (no mass).  Two ways to do each pass:

    exact=False  matrix products, stabilised by the max of V along the summed axis
                 (the fast kernel of the NumPy version; the result is floored at
                 1e-300 times that max).  Needs float64: in float32 exp underflows
                 below e^-87 instead of e^-745 and the kernel breaks down.
    exact=True   exact log-sum-exp (d^3 terms per pass, formed in chunks of at most
                 `chunk` numbers).  Stable for every eps and in float32.
    floor=True   (exact only) applies the same 1e-300 floor as the matrix-product
                 kernel, so a float32 run on the GPU reproduces the float64 one
                 (down to which far-away pixels end up with exactly zero mass).
    """

    def __init__(self, d, eps, device="cpu", dtype=None, exact=False, floor=None, chunk=None):
        _need_torch()
        device = torch.device(device)
        dtype = dtype or torch.float64
        t = np.linspace(0.0, 1.0, d)
        M = -((t[:, None] - t[None, :]) ** 2) / eps          # float64, as in the NumPy code
        if not exact and dtype != torch.float64:
            raise ValueError("the matrix-product kernel needs float64; use exact=True")
        self.exact = exact
        self.floor = (not exact) if floor is None else floor
        self.M = torch.as_tensor(M, dtype=dtype, device=device)
        if not exact:
            self.K1 = torch.as_tensor(np.exp(M), dtype=dtype, device=device)
        self.chunk = chunk or (2 ** 23 if device.type == "cuda" else 2 ** 22)   # numbers per chunk

    def _lse_last(self, X):
        """out[..., i, l] = log sum_j exp(X[..., i, j] + M[l, j])  (sum over the last axis)."""
        if X.is_cuda and _compiled_lse() is not None:
            try:
                return _compiled_lse()(X, self.M)
            except Exception as err:                  # no Triton / compiler: eager chunks
                _disable_compiled_lse(err)
        d = X.shape[-1]
        batch = int(np.prod(X.shape[:-2])) if X.dim() > 2 else 1
        step = max(1, self.chunk // (batch * d * d))
        out = torch.empty(X.shape, dtype=X.dtype, device=X.device)
        for s in range(0, X.shape[-2], step):
            out[..., s:s + step, :] = torch.logsumexp(X[..., s:s + step, None, :] + self.M, dim=-1)
        return out

    def __call__(self, V):
        m1 = V.amax(dim=-2, keepdim=True)                      # max over the summed (row) index
        m1 = torch.where(torch.isfinite(m1), m1, torch.zeros_like(m1))   # columns with no mass
        if not self.exact:                                     # NumPy Kernel, line by line
            A = torch.log(torch.clamp_min(self.K1 @ torch.exp(V - m1), TINY)) + m1
            m2 = A.amax(dim=-1, keepdim=True)
            return torch.log(torch.clamp_min(torch.exp(A - m2) @ self.K1.T, TINY)) + m2
        A = self._lse_last(V.transpose(-1, -2)).transpose(-1, -2)   # sum over the row index
        if self.floor:
            A = torch.maximum(A, m1 + LOG_TINY)
        out = self._lse_last(A)                                     # sum over the column index
        if self.floor:
            out = torch.maximum(out, A.amax(dim=-1, keepdim=True) + LOG_TINY)
        return out


# ----------------------------------------------------------------------------
# Barycenter on a d x d grid with the separable Gibbs kernel
# ----------------------------------------------------------------------------
def grid_barycenter_log(fields, lam, eps, n_iter=600, tol=1e-6, return_err=False,
                        device="auto", dtype="auto"):
    """Sinkhorn barycenter of 2D fields (list of d x d arrays summing to 1).

    Same updates as `barycenter_sinkhorn` in notebooks/barycenters.ipynb:
        ln u_k = ln a_k - ln(K v_k)
        ln b   = sum_k lam_k ln(K^T u_k)       (normalised)
        ln v_k = ln b - ln(K^T u_k)
    Coordinates are rescaled to [0,1]^2, so eps is in those units.
    Runs on `device` (auto: CUDA if available); all fields are updated as one batch.
    """
    dev = pick_device(device)
    dt = pick_dtype(dev, dtype)
    lam = np.asarray(lam, float) / np.sum(lam)
    d = fields[0].shape[0]
    # float64: the matrix-product kernel; float32: exact log-sum-exp with the same floor
    K = LogGibbs(d, eps, dev, dt, exact=dt != torch.float64, floor=True)
    put = lambda x: torch.as_tensor(np.asarray(x), dtype=dt, device=dev)
    logA = put(np.stack([np.log(np.maximum(a, 1e-30)) for a in fields]))
    lam_t = put(lam).view(-1, 1, 1)
    lnv = torch.zeros_like(logA)
    lnb = torch.full((d, d), -2 * np.log(d), dtype=dt, device=dev)
    errs = []
    for it in range(n_iter):
        lnu = logA - K(lnv)
        lnKtu = K(lnu)                              # K symmetric
        new = (lam_t * lnKtu).sum(0)
        new = new - torch.logsumexp(new.reshape(-1), 0)
        lnv = new - lnKtu
        change = (torch.exp(new.double()) - torch.exp(lnb.double())).abs().sum().item()
        errs.append(change)
        lnb = new
        if change < tol and it > 5:
            break
    b = np.exp(lnb.double().cpu().numpy())
    b /= b.sum()
    return (b, np.array(errs)) if return_err else b


# ----------------------------------------------------------------------------
# Images -> measures
# ----------------------------------------------------------------------------
def load_shape(path, size=128, threshold=0.5, margin=0.08, invert=True):
    """Dark silhouette on light background -> probability measure on a grid."""
    im = Image.open(path).convert("L")
    w, h = im.size
    s = max(w, h)
    canvas = Image.new("L", (s, s), 255 if invert else 0)
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    inner = int(round(size * (1 - 2 * margin)))
    small = canvas.resize((inner, inner), Image.LANCZOS)
    arr = np.asarray(small, dtype=float) / 255.0
    if invert:
        arr = 1.0 - arr
    arr = (arr > threshold).astype(float)
    out = np.zeros((size, size))
    o = (size - inner) // 2
    out[o:o + inner, o:o + inner] = arr
    return out / out.sum()


def toy_shape(name, size=128):
    """A few synthetic shapes (Ring, Square, Plus, Diamond, Star, Disk)."""
    t = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(t, t)
    R = np.hypot(X, Y)
    if name == "Ring":
        m = (R < 0.62) & (R > 0.42)
    elif name == "Disk":
        m = R < 0.55
    elif name == "Square":
        m = (np.abs(X) < 0.5) & (np.abs(Y) < 0.5)
    elif name == "Diamond":
        m = np.abs(X) + np.abs(Y) < 0.68
    elif name == "Plus":
        m = ((np.abs(X) < 0.17) & (np.abs(Y) < 0.65)) | ((np.abs(Y) < 0.17) & (np.abs(X) < 0.65))
    elif name == "Star":
        th = np.arctan2(Y, X)
        m = R < 0.40 + 0.25 * np.cos(5 * th + np.pi / 2)
    elif name == "TwoDisks":
        m = (np.hypot(X - 0.45, Y) < 0.28) | (np.hypot(X + 0.45, Y) < 0.28)
    else:
        raise ValueError(name)
    m = m.astype(float)
    return m / m.sum()
