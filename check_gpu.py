"""
check_gpu.py -- check the CUDA setup and measure the speed of the barycenter kernel.

    python check_gpu.py                    # about a minute
    python check_gpu.py --sizes 256 512    # time other grid sizes

1. Prints what PyTorch sees (version, CUDA build, GPU, memory).
2. Computes the same small debiased barycenter (two toy shapes, 96 px) on the CPU in
   float64 (identical to the original NumPy code) and on the GPU in float32, and prints
   the difference.  Expected: a relative difference of about 1e-4 or less.
3. Times one Sinkhorn iteration of photo_barycenter.barycenter (two photos, debiased) at
   a few grid sizes: float32 exact log-sum-exp (compiled with torch.compile, and the
   eager version for comparison) and float64 matrix products.  With --dtype auto the
   barycenter uses the matrix products for eps >= 1e-4 and the exact kernel below.  A barycenter takes
   from a few hundred to a few thousand iterations (print them with --verbose).
"""
import argparse
import time

import numpy as np
import torch

import otlib
import photo_barycenter as pb


def timed(fn, device, reps):
    fn()                                              # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=otlib.DEVICES, default="auto")
    p.add_argument("--sizes", type=int, nargs="+", default=None,
                   help="grid sizes to time (default: 192 256 384 512 on the GPU, 96 192 on the CPU)")
    args = p.parse_args()

    print(f"PyTorch {torch.__version__}, built with CUDA {torch.version.cuda}")
    dev = otlib.pick_device(args.device)
    if dev.type == "cuda":
        pr = torch.cuda.get_device_properties(dev)
        print(f"GPU: {pr.name}, {pr.total_memory / 2**30:.1f} GiB, compute capability {pr.major}.{pr.minor}")

    # 2. same barycenter on the CPU (float64) and on `dev` in float32 (the GPU default)
    N, eps = 96, 6e-4
    fields = [otlib.toy_shape("Ring", N), otlib.toy_shape("Star", N)]
    rng = np.random.default_rng(0)
    cols = [rng.random((N, N, 3)) for _ in fields]
    res = []
    for d, dt in [(torch.device("cpu"), torch.float64), (dev, torch.float32)]:
        t0 = time.perf_counter()
        b, lnu, lnKtu, K, its = pb.barycenter(fields, [0.4, 0.6], eps, device=d, dtype=dt,
                                              return_iterations=True)
        res.append((b, pb.transported_colour(cols, [0.4, 0.6], lnu, lnKtu, K)))
        print(f"barycenter {N} px on {otlib.describe(d, dt)}: {its} iterations, "
              f"{time.perf_counter() - t0:.1f}s")
    (b0, c0), (b1, c1) = res
    heavy = b0 > 0.01 * b0.max()
    print(f"difference: mass max {np.abs(b0 - b1).max() / b0.max():.1e} (relative to the peak), "
          f"L1 {np.abs(b0 - b1).sum():.1e}; colour max {np.abs(c0 - c1).max(-1)[heavy].max():.1e}")

    # 3. time per iteration: K(lnv), K(lnu) on both photos, 3 x K(lnc) (debiasing)
    sizes = args.sizes or ([192, 256, 384, 512] if dev.type == "cuda" else [96, 192])
    print("\ntime per Sinkhorn iteration (two photos, debiased):")
    for N in sizes:
        eps = 5.0 / N ** 2
        line = f"  {N:4d} px"
        modes = [("float32 exact", torch.float32, True, True),
                 ("float32 exact, not compiled", torch.float32, True, False),
                 ("float64 matmul", torch.float64, False, True)]
        for name, dtype, exact, compiled in modes:
            if dev.type == "cpu" and dtype == torch.float32 and (N > 128 or not compiled):
                continue                              # exact kernel on the CPU: too slow to time
            off = otlib._LSE["off"]
            otlib._LSE["off"] = off or not compiled   # the eager kernel, for comparison
            K = otlib.LogGibbs(N, eps, dev, dtype, exact=exact, floor=True)
            V2 = torch.randn(2, N, N, device=dev, dtype=dtype) * 10
            V1 = V2[0]

            def iteration():
                K(V2); K(V2); K(V1); K(V1); K(V1)

            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            reps = max(1, int(2e8 / N ** 3)) if dev.type == "cuda" else 1
            t = timed(iteration, dev, reps)
            mem = f", peak {torch.cuda.max_memory_allocated(dev) / 2**20:.0f} MiB" if dev.type == "cuda" else ""
            otlib._LSE["off"] = off
            line += f"\n      {name:28s} {t * 1e3:8.1f} ms  ({t * 1e3:.0f} s per 1000 iterations{mem})"
        print(line, flush=True)


if __name__ == "__main__":
    main()
