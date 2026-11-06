"""
sw_texture.py -- texture synthesis with a sliced Wasserstein loss, optionally guided by labels.

Method: Heitz, Vanhoey, Chambon & Belcour, "A Sliced Wasserstein Loss for Neural Texture
Synthesis", CVPR 2021 (https://arxiv.org/abs/2006.07229), official code at
https://github.com/tchambon/A-Sliced-Wasserstein-Loss-for-Neural-Texture-Synthesis

The picture x is optimised (L-BFGS on its pixels) so that, at every VGG-19 layer l, the
*distribution* of its feature vectors matches the exemplar's, measured by the sliced
Wasserstein distance:

    SW_2^2(F_x, F_e) = E_theta  W_2^2( <theta, F_x>, <theta, F_e> ),   theta ~ U(S^{C-1})

and each 1-D W_2 is computed exactly by sorting (the 1-D optimal transport is the monotone
map).  With C random directions per layer and per evaluation, as in the paper.

Label guidance (the fox made of smoke): every pixel carries a label.  The loss is computed
label by label -- features of the output's label-k pixels are matched to the features of the
exemplar's label-k pixels only:

    L = sum_l sum_k  (n_k / n) * SW_2^2( F_x[label k], F_e[label k] )

(When the two sets have different sizes, the exemplar's sorted projections are resampled at
the output's quantiles: that is still the exact 1-D optimal transport.)  This is the limit of
the paper's trick of appending the label as an extra, heavily weighted feature channel.

Usage
-----
    pip install torch pillow numpy           (a CUDA build of torch for the GPU)

    # plain texture synthesis, output twice as large as the exemplar
    python sw_texture.py fox_input_texture.png --scale 2

    # the figure: smoke texture guided into the shape of a fox
    python sw_texture.py fox_input_texture.png --labels fox_input_labels.png \
        --target-labels fox_target_labels.png

    # cut-outs with transparency work directly: their alpha is the label map
    # (--size = exemplar resolution, i.e. the scale of the texture; --out-size = output)
    python sw_texture.py snowy_owl_cutout.png --target-labels horse_cutout.png \
        --size 256 --out-size 512 --background white

    # any full-frame photo can fill a shape (examples/ folder; see examples/CREDITS.txt)
    python sw_texture.py examples/blue_smoke.jpg  --target-labels examples/shape_dolphin.png \
        --background white --size 512 --out-size 768 --octaves 2
    # ...but the shape emerges from the texture itself (as in the paper) only when the
    # exemplar has its own natural labels, e.g. glowing gas vs dark space of the nebula:
    python sw_texture.py examples/nebula.jpg --labels examples/nebula_labels.png \
        --target-labels examples/shape_octopus.png --size 512 --out-size 768 --octaves 2
    python sw_texture.py examples/cheetah_fur.jpg --target-labels examples/shape_cat.png \
        --background white --size 512 --out-size 768 --octaves 2

Label maps: black/white masks (thresholded, white = 1), images with a few flat colours (one
label per colour, matched between the two maps), or RGBA cut-outs (opaque = 1, transparent = 0).
A full-frame texture given without --labels fills label 1 of the target; everything else keeps
the --background colour.  The exemplar is then shown to the loss inside random blobs on that
colour, so it also contains texture meeting background (otherwise the shape's edges wash out).

Texture scale = --size / --out-size: a smaller ratio gives finer texture in the shape.
--octaves 2 (or 3) solves at half size first, which keeps large structures at big sizes.

VGG-19 weights: the paper's normalised network (vgg19.pth, 42 MB) is downloaded from the
official repository on the first run and kept next to this file.

Outputs (folder sw_out/ by default): result.png, figure.png (inputs + result side by side),
and progress/iter_XX.png with --save-every.
"""
import argparse
import os
import time
import urllib.request

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
VGG_URL = ("https://raw.githubusercontent.com/tchambon/"
           "A-Sliced-Wasserstein-Loss-for-Neural-Texture-Synthesis/main/vgg19.pth")


# =============================================================================
# VGG-19 as in the paper: reflect padding, average pooling, normalised activations,
# Caffe-style input (BGR, 0..255, mean removed).  Blocks 1-4, every conv layer.
# =============================================================================
class VGG19(torch.nn.Module):
    LAYERS = [("block1_conv1", 3, 64), ("block1_conv2", 64, 64),
              ("block2_conv1", 64, 128), ("block2_conv2", 128, 128),
              ("block3_conv1", 128, 256), ("block3_conv2", 256, 256),
              ("block3_conv3", 256, 256), ("block3_conv4", 256, 256),
              ("block4_conv1", 256, 512), ("block4_conv2", 512, 512),
              ("block4_conv3", 512, 512), ("block4_conv4", 512, 512)]
    POOL_AFTER = {"block1_conv2", "block2_conv2", "block3_conv4"}

    def __init__(self):
        super().__init__()
        for name, cin, cout in self.LAYERS:
            setattr(self, name, torch.nn.Conv2d(cin, cout, 3, padding=1, padding_mode="reflect"))
        self.register_buffer("mean", torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1),
                             persistent=False)

    def forward(self, image):                     # image: (1, 3, H, W), RGB in [0, 1]
        x = 255 * image[:, [2, 1, 0]] - self.mean
        out = []
        for name, _, _ in self.LAYERS:
            x = torch.relu(getattr(self, name)(x))
            out.append(x)
            if name in self.POOL_AFTER:
                x = torch.nn.functional.avg_pool2d(x, 2)
        return out


def load_vgg(device):
    path = os.path.join(HERE, "vgg19.pth")
    if not os.path.exists(path):
        print("downloading the VGG-19 weights of the paper (42 MB) ...", flush=True)
        try:
            urllib.request.urlretrieve(VGG_URL, path + ".part")
            os.replace(path + ".part", path)
        except Exception as err:
            raise SystemExit(f"could not download {VGG_URL}\n  ({err})\n"
                             f"download it by hand and put it next to this script as vgg19.pth")
    vgg = VGG19()
    vgg.load_state_dict(torch.load(path, map_location="cpu"))
    return vgg.to(device).eval().requires_grad_(False)


# =============================================================================
# Images and label maps
# =============================================================================
def load_rgb(path):
    im = Image.open(path)
    return im.convert("RGB"), (np.asarray(im.convert("RGBA"))[..., 3] if has_alpha(im) else None)


def has_alpha(im):
    if im.mode not in ("RGBA", "LA", "PA") and "transparency" not in im.info:
        return False
    a = np.asarray(im.convert("RGBA"))[..., 3]
    return a.min() < 128 <= a.max()


def label_map(path):
    """Integer label map and the 'key' of each label (used to match labels across images)."""
    im = Image.open(path)
    if has_alpha(im):                                           # cut-out: opaque = 1
        return (np.asarray(im.convert("RGBA"))[..., 3] >= 128).astype(np.int64), [0, 1]
    rgb = np.asarray(im.convert("RGB"))
    colours, inv = np.unique(rgb.reshape(-1, 3), axis=0, return_inverse=True)
    grey2 = len(colours) <= 2 and (colours.max(1) == colours.min(1)).all()
    if len(colours) <= 16 and not grey2:                        # flat colours: one label each
        keys = [tuple(int(v) for v in c) for c in colours]
        return inv.reshape(rgb.shape[:2]), keys
    return (rgb.mean(-1) >= 128).astype(np.int64), [0, 1]       # mask: bright = 1


def unify(maps):
    """Re-number labels of several maps so that equal keys get equal ids."""
    keys = sorted({k for _, ks in maps for k in ks}, key=str)
    out = []
    for m, ks in maps:
        lut = np.array([keys.index(k) for k in ks])
        out.append(lut[m])
    return out, len(keys)


def blob_mask(h, w, seed=0, cover=0.65):
    """Random smooth blobs (1) covering `cover` of an h x w canvas."""
    noise = np.random.default_rng(seed).standard_normal((h, w))
    s = max(h, w) / 14                                          # blob size
    fy, fx = np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :]
    smooth = np.real(np.fft.ifft2(np.fft.fft2(noise) * np.exp(-2 * (np.pi * s) ** 2 * (fx ** 2 + fy ** 2))))
    return (smooth > np.quantile(smooth, 1 - cover)).astype(np.int64)


def resize_labels(lab, size):                                   # size = (W, H)
    return np.asarray(Image.fromarray(lab.astype(np.uint8)).resize(size, Image.NEAREST)).astype(np.int64)


def mult8(x):
    return max(8, int(round(x / 8)) * 8)


def to_tensor(im, device):
    return torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)[None].to(device)


def to_image(x):
    return Image.fromarray((x[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                            + 0.5).astype(np.uint8))


# =============================================================================
# Labels at each layer's resolution
# =============================================================================
def layer_labels(lab, n_labels, shape, device):
    """Label of every feature pixel (majority) and how pure the cell is."""
    one_hot = torch.nn.functional.one_hot(torch.from_numpy(lab).to(device), n_labels)
    frac = torch.nn.functional.adaptive_avg_pool2d(one_hot.permute(2, 0, 1)[None].float(), shape)[0]
    purity, ids = frac.max(0)
    return ids.flatten(), purity.flatten()


# =============================================================================
# Sliced Wasserstein loss
# =============================================================================
def resample_sorted(s, n):
    """Sorted rows s (C x m) evaluated at n equally spaced quantiles (C x n)."""
    m = s.shape[1]
    if m == n:
        return s
    pos = torch.linspace(0, m - 1, n, device=s.device)
    lo = pos.floor().long()
    hi = (lo + 1).clamp(max=m - 1)
    w = pos - lo
    return s[:, lo] * (1 - w) + s[:, hi] * w


class SlicedLoss:
    """Labels in `skip` are left out (those pixels are a fixed colour, not synthesised)."""
    def __init__(self, vgg, exemplar, ex_labels, out_labels, n_labels, out_shape, skip=(), purity=0.9):
        self.vgg = vgg
        dev = exemplar.device
        with torch.no_grad():
            acts_e = vgg(exemplar)
            acts_o = vgg(torch.zeros(1, 3, *out_shape, device=dev))     # only for the shapes
        self.ex, self.idx, self.w = [], [], []
        for a_e, a_o in zip(acts_e, acts_o):
            F = a_e[0].flatten(1)                                        # C x n_e
            ids_e, pure = layer_labels(ex_labels, n_labels, a_e.shape[2:], dev)
            ids_o, _ = layer_labels(out_labels, n_labels, a_o.shape[2:], dev)
            ex_l, idx_l, w_l = {}, {}, {}
            for k in ids_o.unique().tolist():
                if k in skip:
                    continue
                sel = (ids_e == k) & (pure >= purity)
                if sel.sum() < 16:                                       # too few pure cells
                    sel = ids_e == k
                if sel.sum() == 0:                                       # region vanished at
                    continue                                             # this layer's size
                ex_l[k] = F[:, sel]
                idx_l[k] = torch.nonzero(ids_o == k).flatten()
                w_l[k] = len(idx_l[k]) / len(ids_o)
            self.ex.append(ex_l); self.idx.append(idx_l); self.w.append(w_l)

    def __call__(self, x):
        loss = 0.0
        for a, ex_l, idx_l, w_l in zip(self.vgg(x), self.ex, self.idx, self.w):
            F = a[0].flatten(1)                                          # C x n_o
            C = F.shape[0]
            dirs = torch.randn(C, C, device=F.device)                    # fresh directions
            dirs = dirs / dirs.norm(dim=1, keepdim=True)
            for k, idx in idx_l.items():
                p_o = torch.sort(dirs @ F[:, idx], dim=1)[0]
                with torch.no_grad():
                    p_e = resample_sorted(torch.sort(dirs @ ex_l[k], dim=1)[0], p_o.shape[1])
                loss = loss + w_l[k] * torch.mean((p_o - p_e) ** 2)
        return loss


# =============================================================================
# Output panel
# =============================================================================
def figure(panels, path, h=320):
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()
    tiles = []
    for title, im in panels:
        im = im.convert("RGB")
        im = im.resize((max(1, round(im.width * h / im.height)), h), Image.LANCZOS)
        tile = Image.new("RGB", (im.width + 16, h + 46), "white")
        tile.paste(im, (8, 38))
        ImageDraw.Draw(tile).text((tile.width // 2, 6), title, fill="black", font=font, anchor="ma")
        tiles.append(tile)
    out = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height), "white")
    x = 0
    for t in tiles:
        out.paste(t, (x, 0)); x += t.width
    out.save(path)


# =============================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("texture", help="exemplar texture (an RGBA cut-out gives its own labels)")
    p.add_argument("--labels", help="label map of the exemplar")
    p.add_argument("--target-labels", help="label map of the output (its shape guides the synthesis)")
    p.add_argument("--size", type=int, default=None,
                   help="resize the exemplar so its longer side is SIZE px (default: as is); "
                        "the target labels are scaled by the same factor unless --out-size is given")
    p.add_argument("--scale", type=float, default=1.0,
                   help="without target labels: output size relative to the exemplar (default 1)")
    p.add_argument("--out-size", type=int, default=None, help="longer side of the output (overrides)")
    p.add_argument("--iters", type=int, default=20, help="outer L-BFGS iterations (default 20)")
    p.add_argument("--lbfgs-steps", type=int, default=64, help="L-BFGS steps per iteration (paper: 64)")
    p.add_argument("--background", choices=["exemplar", "white", "black"], default="exemplar",
                   help="colour of output regions whose label the exemplar lacks, e.g. outside the "
                        "shape for a full-frame texture (default white); for an RGBA exemplar also "
                        "the colour behind the cut-out (default: as stored)")
    p.add_argument("--octaves", type=int, default=1,
                   help="coarse-to-fine: solve at 1/2^(N-1) size first, then refine up to full size; "
                        "2 or 3 helps large structures at sizes >= 384 (default 1, as in the paper)")
    p.add_argument("--out", default=os.path.join(HERE, "sw_out"))
    p.add_argument("--save-every", type=int, default=0, help="save progress/iter_XX.png every N iterations")
    p.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                       "mps" if torch.backends.mps.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)

    # --- exemplar and its labels ---------------------------------------------------------
    tex, alpha = load_rgb(args.texture)
    if alpha is not None and args.background != "exemplar":
        bg = 255 if args.background == "white" else 0
        a = (alpha >= 128)[..., None]
        tex = Image.fromarray(np.where(a, np.asarray(tex), bg).astype(np.uint8))
    W0, H0 = tex.size
    f = args.size / max(W0, H0) if args.size else 1.0
    We, He = mult8(W0 * f), mult8(H0 * f)
    tex = tex.resize((We, He), Image.LANCZOS)

    fill = {"black": 0.0}.get(args.background, 1.0)         # colour of fixed regions
    guided = args.target_labels is not None
    full_frame = guided and not args.labels and alpha is None
    if guided:
        if args.labels:
            ex_map = label_map(args.labels)
        elif alpha is not None:
            ex_map = ((alpha >= 128).astype(np.int64), [0, 1])
        else:
            # full-frame texture: show it inside random blobs on the background colour, so the
            # exemplar also contains texture meeting background (as smoke on white does);
            # otherwise the output's pixels near the shape's edge have nothing to match
            blobs = blob_mask(He, We, args.seed)
            tex = Image.fromarray(np.where(blobs[..., None] == 1, np.asarray(tex),
                                           int(255 * fill)).astype(np.uint8))
            ex_map = (blobs, [0, 1])
        tg_map = label_map(args.target_labels)
        (ex_lab, tg_lab), n_labels = unify([ex_map, tg_map])
        Ht0, Wt0 = tg_lab.shape
        g = args.out_size / max(Wt0, Ht0) if args.out_size else We / W0   # same scale as exemplar
        Wo, Ho = mult8(Wt0 * g), mult8(Ht0 * g)
        ex_lab = resize_labels(ex_lab, (We, He))
        out_lab = resize_labels(tg_lab, (Wo, Ho))
    else:
        g = args.out_size / max(We, He) if args.out_size else args.scale
        Wo, Ho = mult8(We * g), mult8(He * g)
        n_labels = 1
        ex_lab = np.zeros((He, We), np.int64)
        out_lab = np.zeros((Ho, Wo), np.int64)
    # labels of the output that the exemplar does not have (and the background around a
    # full-frame texture): painted in a fixed colour, not synthesised
    fixed = sorted(set(np.unique(out_lab).tolist()) - set(np.unique(ex_lab).tolist())
                   | ({0} if full_frame else set()))
    print(f"device {dev}, exemplar {We}x{He}, output {Wo}x{Ho}, labels {n_labels}"
          + (f", label(s) {fixed} kept {'black' if fill == 0 else 'white'}" if fixed else ""),
          flush=True)

    vgg = load_vgg(dev)
    if args.save_every:
        os.makedirs(os.path.join(args.out, "progress"), exist_ok=True)
    t0 = time.time()
    prev = None
    for level in reversed(range(args.octaves)):          # coarse to fine
        s = 0.5 ** level
        we, he, wo, ho = mult8(We * s), mult8(He * s), mult8(Wo * s), mult8(Ho * s)
        exemplar = to_tensor(tex.resize((we, he), Image.LANCZOS), dev)
        exl, outl = resize_labels(ex_lab, (we, he)), resize_labels(out_lab, (wo, ho))
        loss_fn = SlicedLoss(vgg, exemplar, exl, outl, n_labels, (ho, wo), skip=fixed)
        free = torch.from_numpy(~np.isin(outl, fixed)).to(dev)[None, None].float()   # 1 = synthesised
        if prev is None:            # mean colour of each label + a little noise
            init = torch.full((1, 3, ho, wo), fill, device=dev)
            exl_t, outl_t = torch.from_numpy(exl).to(dev), torch.from_numpy(outl).to(dev)
            for k in outl_t.unique():
                if (exl_t == k).any():
                    init[0][:, outl_t == k] = exemplar[0][:, exl_t == k].mean(1)[:, None]
            init = init + 0.01 * torch.randn_like(init)
        else:                       # previous octave, upsampled
            init = torch.nn.functional.interpolate(prev, size=(ho, wo), mode="bilinear",
                                                   align_corners=False)
        x = torch.nn.Parameter(init)
        compose = lambda: x * free + fill * (1 - free)

        # L-BFGS, as in the paper
        opt = torch.optim.LBFGS([x], lr=1, max_iter=args.lbfgs_steps, tolerance_grad=0.0,
                                tolerance_change=0.0)

        def closure():
            opt.zero_grad()
            loss = loss_fn(compose())
            loss.backward()
            return loss

        tag = f"octave {args.octaves - level}/{args.octaves} " if args.octaves > 1 else ""
        if args.octaves > 1:
            print(f"{tag.strip()}: exemplar {we}x{he}, output {wo}x{ho}", flush=True)
        for it in range(1, args.iters + 1):
            loss = opt.step(closure)
            print(f"  {tag}iteration {it:3d}/{args.iters}  loss {loss.item():.4g}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if args.save_every and it % args.save_every == 0:
                name = (f"oct{args.octaves - level}_" if args.octaves > 1 else "") + f"iter_{it:02d}.png"
                to_image(compose()).save(os.path.join(args.out, "progress", name))
        prev = compose().detach()

    result = to_image(prev)
    result.save(os.path.join(args.out, "result.png"))
    if guided:
        show = lambda lab: Image.fromarray((lab * (255 // max(1, n_labels - 1))).astype(np.uint8))
        figure([("Input labels", show(ex_lab)), ("Input texture", tex),
                ("Target labels", show(out_lab)), ("Synthesis result", result)],
               os.path.join(args.out, "figure.png"))
    else:
        figure([("Input texture", tex), ("Synthesis result", result)], os.path.join(args.out, "figure.png"))
    print(f"done in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
