# How do you average two shapes? — poster code

Code for the figures of the poster *How do you average two shapes? Optimal transport,
Sinkhorn's algorithm and Wasserstein barycenters* (IMPA Tech poster fair, 2026),
by Allan Yukio Furuhashi Melo and Vinícius Flesch Kern.

**The codes in this repository were made with the help of claude opus 5.5**

The algorithms follow the notebooks of the OT course practice sessions
([AI-Quantum-Research-Group/OT-Jan-2026](https://github.com/AI-Quantum-Research-Group/OT-Jan-2026)):
exact OT with POT, log-domain Sinkhorn, and the Sinkhorn barycenter, here in its
debiased version (Janati, Cuturi & Gramfort, ICML 2020).

## Figures

| poster | figure | script | command |
|---|---|---|---|
| block 1 | `fig_rd_barycenter.pdf` — weighted points and their barycenter | `make_figures.py` | `python make_figures.py rd` |
| block 2 | `fig_points.pdf` — optimal coupling, Gaussian cloud → Cristo Redentor | `make_figures.py` | `python make_figures.py points` |
| block 3 | `fig_mse_average.pdf` — average of two bumps, MSE vs $W_2$ | `make_figures.py` | `python make_figures.py avg` |
| block 3 | `fig_color_ot_mse.png` — ferret → octopus colors, OT vs MSE | `color_transfer.py` | `python color_transfer.py` |
| block 4 | `fig_triangle.png` — barycenters of three photos | `barycenter_triangle.py` | `python barycenter_triangle.py` |

Outputs go to `figures/`; intermediate results (recolored photos, cached barycenters)
go to `out/`.

## Files

```
otlib.py                log-domain Sinkhorn, the separable log-domain Gibbs kernel (PyTorch,
                        CUDA or CPU), grid barycenter, image -> measure, --device handling
make_figures.py         the three drawn figures + the shared poster style (colors, fonts, sizes)
color_transfer.py       OT between the color distributions of two photos
photo_barycenter.py     barycenter of photos on a CUDA GPU or the CPU: segmentation (GrabCut),
                        alignment, debiased Sinkhorn barycenter with eps-scaling, colors carried
                        along the transport plan; also makes the cat + owl "meowl" GIF
barycenter_triangle.py  barycenters of three photos for weights on a grid of the simplex
meowl_gpu.py            meowl from cut-outs, any --pair of photos (PHOTOS), on the GPU
                        (CUDA via PyTorch; it used OpenCL before)
check_gpu.py            checks the CUDA setup: GPU vs CPU on a small case, time per iteration
```

## Setup

Python 3.10+. The barycenters run with PyTorch: on an NVIDIA GPU (CUDA) when PyTorch
sees one, otherwise on the CPU. `make_figures.py` and `color_transfer.py` do not need
PyTorch.

**Without an NVIDIA GPU** (e.g. a laptop):

```
pip install -r requirements.txt
```

**On the workstation (NVIDIA RTX 4500, CUDA).** The CUDA toolkit is not needed (the
PyTorch wheels bring their own CUDA libraries), only the NVIDIA driver.

1. Check the driver: `nvidia-smi` must list the RTX 4500. The "CUDA Version" in its
   header is the newest CUDA the driver supports.
2. Install PyTorch with CUDA *first*, then the rest:
   - Linux: `pip install torch` (on Linux the PyPI wheel already includes CUDA;
     PyTorch 2.15 is built for CUDA 13, which needs driver 580 or newer).
   - Windows: PyPI only has the CPU build. Use the command for your system from
     <https://pytorch.org/get-started/locally/>, e.g.
     `pip install torch --index-url https://download.pytorch.org/whl/cu130`.
     The list of CUDA versions changes with each PyTorch release; older indexes such
     as `cu124` only have older PyTorch versions.
   - Driver older than 580 (`nvidia-smi` shows CUDA 12.x): update the driver, or
     install a PyTorch built for CUDA 12, e.g.
     `pip install "torch<2.15" --index-url https://download.pytorch.org/whl/cu128`
     (CUDA 12 builds need driver 525 or newer).

   Then: `pip install -r requirements.txt`.
3. Check: `python check_gpu.py`. It prints the GPU, computes a small barycenter on the GPU
   and on the CPU and prints the difference (expected: about 1e-4 relative or less),
   then times one Sinkhorn iteration at 192–512 px.

## Running on the GPU

The barycenter scripts (`barycenter_triangle.py`, `photo_barycenter.py`, `meowl_gpu.py`)
take `--device auto|cuda|cpu` (default `auto`: the GPU if PyTorch sees one, otherwise
the CPU, with a message saying why) and `--dtype auto|float32|float64`. The commands
of the table above do not change, e.g. `python barycenter_triangle.py`.

|  | GPU (`--device cuda`) | CPU (`--device cpu`) |
|---|---|---|
| precision and kernel (`--dtype auto`) | chosen per eps stage: float64 matrix products while eps >= 1e-4 (the same numbers as the CPU), float32 exact log-sum-exp below | float64, as the original NumPy code: matrix products down to eps = 1e-4, exact log-sum-exp below |
| exact log-sum-exp | compiled with `torch.compile` into one fused kernel (falls back to the plain PyTorch version, with a message, if compiling fails; `OTLIB_NO_COMPILE=1` forces the plain one) | plain PyTorch, in chunks |
| several barycenters / frames | one after another | in parallel processes (`--workers`) |

The algorithm and parameters are the same as before; only where it runs changed. On the
CPU the results are bit-identical to the previous NumPy code; on the GPU they are the same
up to float64 rounding for eps >= 1e-4 (the poster settings). Below eps = 1e-4 the GPU works in float32 and the
barycenters differ by about 1e-4 relative (at most one gray level in the pictures); this
was checked by running the float32 code on a CPU, not on a GPU. Cached results
(`out/triangle/`, `out/meowl/frames/`) keep their names and are reused on either device:
delete them to recompute. An explicit `--dtype float32` or `--dtype float64` uses that
precision at every stage (float32 is then always the exact kernel).

`barycenter_triangle.py` prints its progress (one line per eps stage and every 500
iterations of the last one, with the elapsed time); `--quiet` turns it off. The first
exact-kernel stage starts with a pause of a few tens of seconds while `torch.compile`
builds the kernel.

`meowl_gpu.py` is kept (and now runs on CUDA through PyTorch instead of OpenCL) because
it has the cut-out photos, `--pair` and `--bg`, and its own settings (exact kernel at
every eps, earlier stop when float32 stalls). It shares the iterations with
`photo_barycenter.py`, which also runs on the GPU now. `--list-devices` lists the CUDA
GPUs, and `--exact-exp` (OpenCL only) is gone.

**Larger grids and smaller eps.** GPU memory is not the limit (`check_gpu.py` prints
the peak); time is: one iteration costs about size³, and a smaller eps needs more
iterations (the last stage can run up to 20000). The poster figures use the defaults, so keep them to reproduce the poster;
for sharper pictures on the workstation, for example:

```
python barycenter_triangle.py --size 384 --eps 3.5e-5   # keep eps * size^2 >= ~5
python photo_barycenter.py --size 384 --eps 4e-5        # keep eps >= 2 / size^2
python meowl_gpu.py --size 512 --eps 2e-5 --steps 8 --verbose
```

## Input data (`data/`, not in the repository)

The input images are third-party, so they are not committed. Put them in `data/`:

| file | used by | where it comes from |
|---|---|---|
| `cristo_redentor.jpg` | `fig_points` | `imgs/` of the [course repository](https://github.com/AI-Quantum-Research-Group/OT-Jan-2026) (icon by katyam1983112083 / Vecteezy) |
| `ferret.png`, `octopus.png` | `fig_color_ot_mse` | any two photos (any size) |
| `cat_photo.png`, `owl_photo.png` | `photo_barycenter.py`, `fig_triangle` | photos; the GrabCut rectangle and eye positions are in `PHOTOS` in `photo_barycenter.py` |
| `snowy_to_brown_strip.png` | `fig_triangle` | `strip.png` from `python meowl_gpu.py --pair snowy_owl brown_owl --steps 11` |
| `*_cutout.png` | `meowl_gpu.py` | cut-outs with a transparent background (see `PHOTOS` in `meowl_gpu.py`) |

To use other photos, change the file names (and, for the barycenters, the eye positions
in `PHOTOS`).

## Running time

On a laptop CPU: `make_figures.py` and `color_transfer.py` take about 1–2 minutes;
`barycenter_triangle.py` about 5–10 minutes (it uses all cores and caches each
barycenter in `out/triangle/`, so a second run only redraws the figure).
`photo_barycenter.py` at the default 256 px takes a few minutes per frame; for smaller
`--eps` it switches to an exact (much slower) kernel, which is where a GPU helps most.

On the GPU these are rough expectations, not measurements (the port was written on a
machine without a GPU; `python check_gpu.py` measures the time per iteration on yours).
A barycenter takes a few hundred to a few thousand iterations, and one iteration costs
about size³ operations. Expect seconds to a couple of minutes per barycenter at the
default sizes (192 px for the triangle, 256 px per meowl frame), and about 8 times as
long at twice the size (several minutes per frame at 512 px). The scripts print the time
of every barycenter, and `--verbose` prints the iterations.

## References

1. AI & Quantum Research Group (A. Gerolin et al., University of Ottawa).
   *Optimal Transport: practice sessions*, 2026.
   https://github.com/AI-Quantum-Research-Group/OT-Jan-2026
2. H. Janati, M. Cuturi, A. Gramfort. *Debiased Sinkhorn barycenters*. ICML 2020.
3. R. Flamary et al. *POT: Python Optimal Transport*. JMLR 22(78), 2021.
