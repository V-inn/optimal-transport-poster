"""
otlib.py -- small NumPy toolbox used to generate the poster figures.

It follows the same algorithms as the notebooks in
https://github.com/AI-Quantum-Research-Group/OT-Jan-2026 (intro, sinkhorn,
barycenters), written in plain NumPy so that every figure can be reproduced
without JAX / uot-bench:

  * sinkhorn_log          -- log-domain Sinkhorn for entropic OT (dense cost)
  * grid_barycenter_log   -- the log-domain Sinkhorn barycenter of
                             notebooks/barycenters.ipynb, on a d x d image grid,
                             using the fact that exp(-|x-y|^2/eps) is separable
                             (K = K1 (x) K1), so K v costs two matrix products.
                             This is what makes 128x128 shapes feasible.
  * load_shape            -- image -> probability measure on a grid
                             (same idea as uot.data.load_image_as_binary_grid)
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from PIL import Image

TINY = 1e-300


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
# Barycenter on a d x d grid with the separable Gibbs kernel
# ----------------------------------------------------------------------------
def _gibbs_1d(d, eps):
    t = np.linspace(0.0, 1.0, d)
    return np.exp(-((t[:, None] - t[None, :]) ** 2) / eps)


def _log_conv(lnV, K1):
    """log( K * exp(lnV) ) for K = K1 (x) K1, stabilised column/row-wise."""
    m1 = lnV.max(axis=0, keepdims=True)                       # (1, d)
    A = np.log(np.maximum(K1 @ np.exp(lnV - m1), TINY)) + m1  # sum over first index
    m2 = A.max(axis=1, keepdims=True)                         # (d, 1)
    return np.log(np.maximum(np.exp(A - m2) @ K1.T, TINY)) + m2


def grid_barycenter_log(fields, lam, eps, n_iter=600, tol=1e-6, return_err=False):
    """Sinkhorn barycenter of 2D fields (list of d x d arrays summing to 1).

    Same updates as `barycenter_sinkhorn` in notebooks/barycenters.ipynb:
        ln u_k = ln a_k - ln(K v_k)
        ln b   = sum_k lam_k ln(K^T u_k)       (normalised)
        ln v_k = ln b - ln(K^T u_k)
    Coordinates are rescaled to [0,1]^2, so eps is in those units.
    """
    lam = np.asarray(lam, float) / np.sum(lam)
    d = fields[0].shape[0]
    K1 = _gibbs_1d(d, eps)
    logA = [np.log(np.maximum(a, 1e-30)) for a in fields]
    lnv = [np.zeros((d, d)) for _ in fields]
    lnb = np.full((d, d), -2 * np.log(d))
    errs = []
    for it in range(n_iter):
        lnKtu = []
        for k in range(len(fields)):
            lnu = logA[k] - _log_conv(lnv[k], K1)
            lnKtu.append(_log_conv(lnu, K1))       # K symmetric
        new = sum(l * x for l, x in zip(lam, lnKtu))
        new -= logsumexp(new)
        lnv = [new - x for x in lnKtu]
        change = np.abs(np.exp(new) - np.exp(lnb)).sum()
        errs.append(change)
        lnb = new
        if change < tol and it > 5:
            break
    b = np.exp(lnb)
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
