"""Three-row reconstruction figure as a PDF, on white, for dropping into the paper.

Rows, top to bottom:
  1  our predicted tactile normal video
  2  the 3D surface relief obtained by integrating those normals into a heightmap
  3  the colour image a camera-based tactile sensor would have measured, simulated
     from that same heightmap

Columns are frames of one press. There is no reference row: what the prediction was
transferred from belongs in the caption, not in the figure.

Differences from 03_recon_visuotactile/make_recon_figure.py, which stitches a preview
with OpenCV: this lays the page out in inches, leaves white gaps between frames, writes
its labels as real text in black on white (so the PDF can be copied into a document and
still have selectable text), and lets the relief's height axis be squashed with
--z_scale so the surface reads as a shallow relief rather than a mountain range.

    python make_recon_pdf.py --obj 993 --pair 2
    python make_recon_pdf.py --obj 994 --pair 3 --z_scale 0.14
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
sys.path.insert(0, f"{ROOT}/paper_experiments/03_recon_visuotactile")
sys.path.insert(0, f"{ROOT}/paper_experiments/04_paper_figures")
sys.path.insert(0, f"{ROOT}/rebot_net")

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import cm                            # noqa: E402
from scipy.ndimage import gaussian_filter            # noqa: E402

from make_recon_figure import (FLAT, H_, W_, normal_to_height, orient_up,   # noqa: E402
                               taxim_rgb)
from figlib import Page                              # noqa: E402
from dataset import TactileTransferDataset           # noqa: E402
from train import build_model                        # noqa: E402

TRANSFER = f"{ROOT}/log/transfer_feat_match_pseudo_mini_tactile_normal_superpoint_superglue_normalmatch"
COND = f"{ROOT}/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini"
CKPT = f"{ROOT}/log/rebot_checkpoints_S_geomcat_film/best.pth"
OUT = f"{ROOT}/log/paper_job05_recon_figure"

ROWS = [("pred", "Predicted tactile normal (ours)"),
        ("relief", "3D surface reconstructed from the prediction"),
        ("rgb", "Simulated colour image of a camera-based tactile sensor")]


def render_relief(H, out_path, z_scale=0.16, light_gain=None, size=(4.2, 3.15), dpi=150,
                  bg="black", contrast=1.35, ambient=0.10):
    """Shaded relief of a heightmap, with a squashable height axis.

    z_scale sets how tall the surface is drawn relative to its width. The shading
    is computed from the same exaggeration, so a flattened surface also looks
    flatter rather than keeping the shadows of a tall one.

    Flattening the surface also flattens its shading, which left the relief a
    narrow band of greys. contrast stretches the shading about its mid-tone and
    ambient sets how dark the darkest facets are allowed to go, so the surface
    keeps its range once squashed.

    The relief sits on black by default: the grey surface reads far better against
    it than against the white of the page, which is how the earlier version of
    this figure was drawn. Everything else on the page stays white with black
    text.
    """
    if light_gain is None:
        # Shading strength is kept at the value the original (un-flattened) figure
        # used. Scaling it down with z_scale did keep shading and geometry
        # consistent, but it left the squashed surface a narrow band of greys.
        light_gain = 9.0
    Hn = H - H.min()
    Hn = Hn / (Hn.max() + 1e-8)
    Hn = gaussian_filter(Hn, sigma=2.6)      # smooth the integration stepping
    Hn = Hn[::2, ::2]                        # lighter mesh
    rows, cols = Hn.shape
    Y, X = np.mgrid[0:rows, 0:cols]
    zy, zx = np.gradient(Hn * light_gain)
    nrm = np.dstack([-zx, -zy, np.ones_like(Hn)])
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    lgt = np.array([-0.5, -0.6, 0.7])
    lgt /= np.linalg.norm(lgt)
    inten = np.clip((nrm * lgt).sum(2), 0, 1)
    inten = np.clip(0.5 + contrast * (inten - 0.5), 0, 1)
    shaded = cm.gray(ambient + (1.0 - ambient) * inten)

    fig = plt.figure(figsize=size, dpi=dpi, facecolor=bg)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(bg)
    ax.plot_surface(X, Y, Hn, facecolors=shaded, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    ax.set_axis_off()
    ax.view_init(elev=52, azim=-62)
    ax.set_box_aspect((cols, rows, z_scale * max(rows, cols)))
    ax.set_zlim(0, 1)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, dpi=dpi, facecolor=bg, pad_inches=0)
    plt.close(fig)
    im = cv2.cvtColor(cv2.imread(out_path), cv2.COLOR_BGR2RGB)
    im = fit_4x3(crop_to_content(im, bg=bg), bg=bg)
    cv2.imwrite(out_path, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    return im


def crop_to_content(im, thr=250, pad=6, bg="white"):
    """Trim the empty margin a 3D axes leaves around the surface."""
    ink = (im.max(axis=2) > 255 - thr) if bg == "black" else (im.min(axis=2) < thr)
    if not ink.any():
        return im
    ys, xs = np.where(ink)
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, im.shape[0])
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, im.shape[1])
    return im[y0:y1, x0:x1]


def fit_4x3(im, bg="white"):
    """Pad to the same 4:3 shape as the tactile frames, so the columns of the
    figure line up."""
    h, w = im.shape[:2]
    tw, th = (w, int(round(w * 3 / 4))) if w * 3 / 4 >= h else (int(round(h * 4 / 3)), h)
    out = np.full((th, tw, 3), 0 if bg == "black" else 255, np.uint8)
    y0, x0 = (th - h) // 2, (tw - w) // 2
    out[y0:y0 + h, x0:x0 + w] = im
    return out


def predict(obj, pair, transfer_dir, cond_dir, layout):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model("rebot_S", cond_chans=3, film_chans=0, bottleneck_hw=24,
                        time_cond="film").to(device)
    ck = torch.load(CKPT, map_location=device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    class Flexible(TactileTransferDataset):
        def __init__(self, *a, **k):
            self.NUM_PAIRS = 32
            super().__init__(*a, **k)

        def _obj_dir(self, obj_id):
            base = os.path.join(self.transfer_dir, str(obj_id))
            return os.path.join(base, "transfer") if layout == "nested" else base

    ds = Flexible(transfer_dir, [obj], split="test", cond_dir=cond_dir,
                  film_modality="normal", film_scale=100, geom_concat=True,
                  video_type="tactile_normal", time_cond="film")
    if not ds.lq_video_exists(obj, pair):
        raise SystemExit(f"no transferred video for object {obj}, touch {pair} "
                         f"under {transfer_dir}")
    preds, gts = [], []
    with torch.no_grad():
        for lq, gt, blank, film, t_norm in ds.iter_video_pairs(obj, pair):
            t_in = torch.tensor([t_norm], device=device)
            pr = model(lq.unsqueeze(0).to(device), film=None, t=t_in).squeeze(0)
            preds.append(pr.cpu().clamp(0, 1).permute(1, 2, 0).numpy())
            gts.append(gt.permute(1, 2, 0).numpy())
    return preds, gts, ck.get("epoch")


def contact_frames(gts, n_cols):
    """Evenly spaced frames over the part of the press that is actually in contact.

    The first and last frames of a press are flat no-contact readings; integrating
    those only amplifies compression noise, so they are left out.
    """
    dev = np.array([np.linalg.norm(2 * g - 1 - FLAT, axis=-1).mean() for g in gts])
    contact = np.where(dev > dev.min() + 0.3 * (dev.max() - dev.min()))[0]
    lo, hi = ((int(contact[0]), int(contact[-1])) if len(contact) >= n_cols
              else (0, len(gts) - 1))
    return np.linspace(lo, hi, n_cols).round().astype(int).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", type=int, required=True)
    ap.add_argument("--pair", type=int, required=True)
    ap.add_argument("--n_cols", type=int, default=6)
    ap.add_argument("--z_scale", type=float, default=0.16,
                    help="height of the 3D relief relative to its width; the old "
                         "figure used 0.42, lower is flatter")
    ap.add_argument("--width", type=float, default=7.0, help="page width in inches")
    ap.add_argument("--gap", type=float, default=0.055, help="white gap between frames")
    ap.add_argument("--bottom", type=float, default=0.24,
                    help="white space left below the last row, in inches")
    ap.add_argument("--relief_light", type=float, default=None,
                    help="shading strength of the relief (default 9.0, the value the "
                         "un-flattened figure used)")
    ap.add_argument("--relief_contrast", type=float, default=1.35,
                    help="how far the relief's shading is stretched about its mid-tone; "
                         "1.0 is the unstretched shading")
    ap.add_argument("--relief_ambient", type=float, default=0.10,
                    help="how dark the darkest facets of the relief may go")
    ap.add_argument("--relief_bg", default="black", choices=["black", "white"],
                    help="background behind the 3D relief only; the page itself stays white")
    ap.add_argument("--transfer_dir", default=TRANSFER)
    ap.add_argument("--cond_dir", default=COND)
    ap.add_argument("--layout", default="flat", choices=["flat", "nested"])
    ap.add_argument("--out_dir", default=OUT)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or f"{args.obj}_{args.pair}"
    cells = f"{args.out_dir}/assets_pdf/{tag}"
    os.makedirs(cells, exist_ok=True)

    preds, gts, epoch = predict(args.obj, args.pair, args.transfer_dir,
                                args.cond_dir, args.layout)
    idxs = contact_frames(gts, args.n_cols)
    print(f"object {args.obj} touch {args.pair}: {len(preds)} frames, columns at {idxs}")

    imgs = {k: [] for k, _ in ROWS}
    for j, t in enumerate(idxs):
        pred = preds[t]
        mask = np.linalg.norm(2 * gts[t] - 1 - FLAT, axis=-1) > 0.15
        Hs = orient_up(normal_to_height(pred, out_hw=(H_, W_)),
                       cv2.resize(mask.astype(np.uint8), (W_, H_)).astype(bool))

        p_pred = f"{cells}/col{j:02d}_f{t:03d}_row1_prediction.png"
        p_rel = f"{cells}/col{j:02d}_f{t:03d}_row2_relief.png"
        p_rgb = f"{cells}/col{j:02d}_f{t:03d}_row3_simulated_rgb.png"
        cv2.imwrite(p_pred, cv2.cvtColor((np.clip(pred, 0, 1) * 255).astype(np.uint8),
                                         cv2.COLOR_RGB2BGR))
        relief = render_relief(Hs, p_rel, z_scale=args.z_scale, bg=args.relief_bg,
                               light_gain=args.relief_light,
                               contrast=args.relief_contrast, ambient=args.relief_ambient)
        rgb = taxim_rgb(Hs)
        cv2.imwrite(p_rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        imgs["pred"].append((np.clip(pred, 0, 1) * 255).astype(np.uint8))
        imgs["relief"].append(relief)
        imgs["rgb"].append(rgb)
        print(f"  column {j} (frame {t}) done", flush=True)

    # ---- page layout, in inches -------------------------------------------
    n = len(idxs)
    left, right = 0.09, 0.09
    iw = (args.width - left - right - (n - 1) * args.gap) / n
    ih = iw * 0.75
    label_h, row_gap = 0.155, 0.10
    top = 0.28
    # the last row used to run right into the bottom edge of the page
    height = top + 3 * (label_h + ih) + 2 * row_gap + args.bottom

    p = Page(args.width, height)
    p.text(left, 0.14, f"object {args.obj}, touch {args.pair}", size=5.6, color="0.35")
    p.text(args.width - right - 0.44, 0.14, "time", size=5.6, ha="right", color="0.45",
           style="italic")
    p.arrow(args.width - right - 0.40, 0.14, args.width - right - 0.02, 0.14,
            color="0.55", lw=0.8, mut=4)

    y = top
    for k, (key, label) in enumerate(ROWS):
        p.text(left, y + 0.06, label, size=6.4, color="black")
        yy = y + label_h
        for j, im in enumerate(imgs[key]):
            # no border: the frames sit directly on the white page
            p.img(left + j * (iw + args.gap), yy, iw, ih, im, lw=0)
        y = yy + ih + row_gap

    stem = f"{args.out_dir}/recon_{tag}"
    p.save(stem)
    print(f"cells: {cells}")


if __name__ == "__main__":
    main()
