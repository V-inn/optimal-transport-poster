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
otlib.py                log-domain Sinkhorn, separable grid barycenter, image -> measure
make_figures.py         the three drawn figures + the shared poster style (colors, fonts, sizes)
color_transfer.py       OT between the color distributions of two photos
photo_barycenter.py     barycenter of photos on the CPU: segmentation (GrabCut), alignment,
                        debiased Sinkhorn barycenter with eps-scaling, colors carried along
                        the transport plan; also makes the cat + owl "meowl" GIF
barycenter_triangle.py  barycenters of three photos for weights on a grid of the simplex
meowl_gpu.py            same as photo_barycenter.py, on the GPU (OpenCL, float32)
```

## Setup

```
pip install -r requirements.txt
```

Python 3.10+. `pyopencl` is only needed for `meowl_gpu.py`.

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
`--eps` it switches to an exact (slower) kernel — `meowl_gpu.py` is the fast option.

## References

1. AI & Quantum Research Group (A. Gerolin et al., University of Ottawa).
   *Optimal Transport: practice sessions*, 2026.
   https://github.com/AI-Quantum-Research-Group/OT-Jan-2026
2. H. Janati, M. Cuturi, A. Gramfort. *Debiased Sinkhorn barycenters*. ICML 2020.
3. R. Flamary et al. *POT: Python Optimal Transport*. JMLR 22(78), 2021.
