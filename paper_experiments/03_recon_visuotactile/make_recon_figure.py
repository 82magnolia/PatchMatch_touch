"""3D surface reconstruction + visuo-tactile sensor simulation figure.

For a touch taken from the ground-truth-retrieval benchmark, this builds a
four-row strip in which every column is one frame of the touch:

  row 1  reference tactile normal video          (the example we transfer from)
  row 2  predicted tactile normal video          (refinement-network output)
  row 3  hillshaded 3D relief of the heightmap   (Poisson integration of row 2)
  row 4  RGB visuo-tactile frames                (Taxim optical simulation of row 3)

Each cell is saved as a separate PNG under the asset directory so the figure can
be re-laid-out in LaTeX, and a stitched preview PNG plus an MP4 of the full
sequence are written alongside.
"""
import argparse
import os
import subprocess
import sys

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm      # noqa: E402

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
sys.path.insert(0, f"{ROOT}/rebot_net")
sys.path.insert(0, f"{ROOT}/Taxim")
sys.path.insert(0, f"{ROOT}/baselines/RandomQuiltingTactile/TactileDreamFusion")
from dataset import TactileTransferDataset          # noqa: E402
from train import build_model                       # noqa: E402
from poisson_solver import poisson_dct_neumann      # noqa: E402
from Basics.CalibData import CalibData              # noqa: E402
import Basics.sensorParams as psp                   # noqa: E402
import Basics.params as pr                          # noqa: E402
from scipy.ndimage import gaussian_filter           # noqa: E402

TRANSFER = f"{ROOT}/log/transfer_feat_match_pseudo_mini_tactile_normal_superpoint_superglue"
COND = f"{ROOT}/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini"
CKPT = f"{ROOT}/log/rebot_checkpoints_S_geomcat_film/best.pth"
CAL = f"{ROOT}/Taxim/calibs"

H_, W_ = psp.h, psp.w
FLAT = np.array([0.0, 0.0, 1.0])

# ---------------------------------------------------------------- Taxim optics
_calib = CalibData(f"{CAL}/polycalib.npz")
_bins = int(_calib.numBins)
_f0 = np.load(f"{CAL}/dataPack.npz", allow_pickle=True)["f0"]
if _f0.shape[:2] != (H_, W_):
    _f0 = cv2.resize(_f0, (W_, H_))
_xx, _yy = np.meshgrid(range(W_), range(H_))
_A = np.array([_xx.flatten()**2, _yy.flatten()**2, _xx.flatten()*_yy.flatten(),
               _xx.flatten(), _yy.flatten(), np.ones(H_*W_)]).T


def _proc_bg(f0):
    img = f0.astype(float)
    sm = np.stack([gaussian_filter(img[:, :, c], pr.kscale) for c in range(3)], -1)
    idx = np.mean(sm - img, axis=2) < pr.diffThreshold
    p = pr.frameMixingPercentage
    for c in range(3):
        sm[:, :, c][idx] = p * sm[:, :, c][idx] + (1 - p) * img[:, :, c][idx]
    return sm


_BG = _proc_bg(_f0)


def _gen_normals(H):
    h, w = H.shape
    dzdx = (H[2:h, 1:w-1] - H[0:h-2, 1:w-1]) / 2.0
    dzdy = (H[1:h-1, 2:w] - H[1:h-1, 0:w-2]) / 2.0
    mag = np.sqrt(dzdx**2 + dzdy**2)
    gm = np.arctan(mag)
    gd = np.zeros_like(mag)
    v = mag != 0
    gd[v] = np.arctan2(dzdx[v] / mag[v], dzdy[v] / mag[v])
    return np.pad(gm, 1, mode="edge"), np.pad(gd, 1, mode="edge")


def taxim_rgb(H):
    """Heightmap at sensor resolution -> simulated GelSight RGB image."""
    gm, gd = _gen_normals(H)
    ix = np.clip(np.floor(gm / (0.5 * np.pi / (_bins - 1))).astype(int), 0, _bins - 1)
    iy = np.clip(np.floor((gd + np.pi) / (2 * np.pi / (_bins - 1))).astype(int), 0, _bins - 1)
    est = np.zeros((H_, W_, 3))
    for c, g in enumerate([_calib.grad_r, _calib.grad_g, _calib.grad_b]):
        pm = g[ix, iy, :].reshape(H_ * W_, g.shape[2])
        est[:, :, c] = np.sum(_A * pm, axis=1).reshape(H_, W_)
    return np.clip(est + _BG, 0, 255).astype(np.uint8)


# ------------------------------------------------------- normal -> height
def normal_to_height(rgb01, out_hw=None):
    """Colour-coded normal map -> heightmap, with the global tilt removed."""
    if out_hw is not None:
        rgb01 = cv2.resize(rgb01, (out_hw[1], out_hw[0]))
    n = 2.0 * rgb01 - 1.0
    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
    nz = np.clip(n[..., 2], 0.05, 1.0)
    H = poisson_dct_neumann(-n[..., 0] / nz, -n[..., 1] / nz)
    r, c = np.mgrid[0:H.shape[0], 0:H.shape[1]]
    A = np.c_[r.ravel(), c.ravel(), np.ones(H.size)]
    coef, *_ = np.linalg.lstsq(A, H.ravel(), rcond=None)
    return H - (A @ coef).reshape(H.shape)


def orient_up(H, mask):
    if mask.sum() < 10:
        return H
    if H[mask].mean() < H[~mask].mean():
        H = -H
    return H


def render_3d(H, out_path):
    """Hillshaded 3D relief of the heightmap.

    Ported verbatim in spirit from train_refine_scripts/time_cond_sweep/
    height3d_geomcat_film.py: direct Lambertian (diffuse) shading computed from
    the surface normals, which gives a smooth matte relief and avoids the
    contour rings that matplotlib's LightSource hillshade produces on these
    lightly-banded (video-compressed) normals.
    """
    Hn = H - H.min()
    Hn = Hn / (Hn.max() + 1e-8)
    Hn = gaussian_filter(Hn, sigma=2.6)      # smooth the integration stepping
    ds = 2                                    # lighter mesh
    Hn = Hn[::ds, ::ds]
    rows, cols = Hn.shape
    Y, X = np.mgrid[0:rows, 0:cols]
    zy, zx = np.gradient(Hn * 9.0)
    nrm = np.dstack([-zx, -zy, np.ones_like(Hn)])
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    lgt = np.array([-0.5, -0.6, 0.7])
    lgt /= np.linalg.norm(lgt)
    inten = np.clip((nrm * lgt).sum(2), 0, 1)
    inten = 0.28 + 0.72 * inten              # ambient + diffuse
    shaded = cm.gray(inten)
    fig = plt.figure(figsize=(4.2, 4.2), facecolor="black")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")
    ax.plot_surface(X, Y, Hn, facecolors=shaded, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    ax.set_axis_off()
    ax.view_init(elev=55, azim=-62)
    ax.set_box_aspect((cols, rows, 0.42 * max(rows, cols)))
    ax.set_zlim(0, 1)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, dpi=110, facecolor="black", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return cv2.imread(out_path)


def save_rgb(path, rgb_u8):
    cv2.imwrite(path, cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))


def label_strip(img_bgr, text, h=28):
    bar = np.zeros((h, img_bgr.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img_bgr])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", type=int, required=True)
    ap.add_argument("--pair", type=int, required=True)
    ap.add_argument("--n_cols", type=int, default=6,
                    help="How many frames of the touch to show as columns.")
    ap.add_argument("--out_dir", default=f"{ROOT}/log/paper_job03_recon_visuotactile")
    ap.add_argument("--transfer_dir", default=TRANSFER,
                    help="root of the coarse transfers to refine (default: the "
                         "ground-truth-retrieval run)")
    ap.add_argument("--cond_dir", default=COND,
                    help="root holding {obj}/{idx}_scale100_normal.jpg for the query")
    ap.add_argument("--layout", default="flat", choices=["flat", "nested"],
                    help="'flat' is {transfer_dir}/{obj}/, 'nested' is "
                         "{transfer_dir}/{obj}/transfer/ as transfer_pipeline.py writes it")
    ap.add_argument("--tag", default=None,
                    help="name for the output files (default: {obj}_{pair})")
    ap.add_argument("--video", action="store_true", help="Also write the full-length MP4.")
    args = ap.parse_args()

    tag = args.tag or f"{args.obj}_{args.pair}"
    asset = os.path.join(args.out_dir, "assets", tag)
    os.makedirs(asset, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model("rebot_S", cond_chans=3, film_chans=0,
                        bottleneck_hw=24, time_cond="film").to(device)
    ck = torch.load(CKPT, map_location=device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    print(f"loaded refinement net @ epoch {ck.get('epoch')}")

    class Flexible(TactileTransferDataset):
        """Same dataset, but able to read transfer_pipeline.py's nested layout."""

        def __init__(self, *a, **k):
            self.NUM_PAIRS = 32
            super().__init__(*a, **k)

        def _obj_dir(self, obj_id):
            base = os.path.join(self.transfer_dir, str(obj_id))
            return os.path.join(base, "transfer") if args.layout == "nested" else base

    ds = Flexible(args.transfer_dir, [args.obj], split="test", cond_dir=args.cond_dir,
                  film_modality="normal", film_scale=100,
                  geom_concat=True, video_type="tactile_normal",
                  time_cond="film")
    if not ds.lq_video_exists(args.obj, args.pair):
        raise SystemExit(f"no transferred video for {args.obj}_{args.pair}")

    preds, gts = [], []
    with torch.no_grad():
        for lq, gt, blank, film, t_norm in ds.iter_video_pairs(args.obj, args.pair):
            t_in = torch.tensor([t_norm], device=device)
            pr = model(lq.unsqueeze(0).to(device), film=None, t=t_in).squeeze(0)
            preds.append(pr.cpu().clamp(0, 1).permute(1, 2, 0).numpy())
            gts.append(gt.permute(1, 2, 0).numpy())
    n = len(preds)
    print(f"{n} frames")

    # Reference video frames (the example the prediction was transferred from)
    ref_path = os.path.join(ds._obj_dir(args.obj), f"{args.pair}_ref_tactile_normal.mp4")
    cap = cv2.VideoCapture(ref_path)
    refs = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        refs.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    cap.release()

    # Columns: evenly spaced frames covering the part of the press cycle where
    # the sensor is actually in contact. The first and last frames of a
    # back_forth_press touch are flat no-contact readings; integrating those to
    # a heightmap only amplifies compression noise, so they are excluded.
    # Thresholding on the *range* of the deviation, not its maximum: on a curved
    # object the untouched gel already reads far from flat, so a fraction-of-max
    # threshold would mark every frame as contact.
    dev = np.array([np.linalg.norm(2 * g - 1 - FLAT, axis=-1).mean() for g in gts])
    contact = np.where(dev > dev.min() + 0.3 * (dev.max() - dev.min()))[0]
    lo, hi = (int(contact[0]), int(contact[-1])) if len(contact) >= args.n_cols else (0, n - 1)
    idxs = np.linspace(lo, hi, args.n_cols).round().astype(int).tolist()
    print("columns at frames", idxs)

    rows = {"ref": [], "pred": [], "cloud": [], "rgb": []}
    for j, t in enumerate(idxs):
        pred = preds[t]
        gtn = gts[t]
        mask = np.linalg.norm(2 * gtn - 1 - FLAT, axis=-1) > 0.15
        Hs = orient_up(normal_to_height(pred, out_hw=(H_, W_)),
                       cv2.resize(mask.astype(np.uint8), (W_, H_)).astype(bool))

        p_ref = f"{asset}/col{j:02d}_f{t:03d}_row1_ref_normal.png"
        p_pred = f"{asset}/col{j:02d}_f{t:03d}_row2_pred_normal.png"
        p_cloud = f"{asset}/col{j:02d}_f{t:03d}_row3_height3d.png"
        p_rgb = f"{asset}/col{j:02d}_f{t:03d}_row4_taxim_rgb.png"

        ref_img = (np.clip(refs[min(t, len(refs) - 1)], 0, 1) * 255).astype(np.uint8) \
            if refs else np.zeros((240, 320, 3), np.uint8)
        save_rgb(p_ref, ref_img)
        save_rgb(p_pred, (np.clip(pred, 0, 1) * 255).astype(np.uint8))
        cloud = render_3d(Hs, p_cloud)
        rgb = taxim_rgb(Hs)
        save_rgb(p_rgb, rgb)

        rows["ref"].append(cv2.imread(p_ref))
        rows["pred"].append(cv2.imread(p_pred))
        rows["cloud"].append(cloud)
        rows["rgb"].append(cv2.imread(p_rgb))
        print(f"  col {j} (frame {t}) done", flush=True)

    # ---- stitched preview -------------------------------------------------
    CH, CW = 210, 280
    def strip(imgs):
        return np.hstack([cv2.resize(im, (CW, CH)) for im in imgs])

    labels = ["Reference tactile normal (the example)",
              "Predicted tactile normal (our refinement network)",
              "3D reconstruction from the predicted heightmap",
              "Simulated RGB visuo-tactile frames (Taxim)"]
    blocks = [label_strip(strip(rows[k]), lab)
              for k, lab in zip(("ref", "pred", "cloud", "rgb"), labels)]
    grid = np.vstack(blocks)
    hdr = np.zeros((30, grid.shape[1], 3), np.uint8)
    cv2.putText(hdr, f"object {args.obj}, touch {args.pair} - frames "
                     f"{idxs[0]}..{idxs[-1]} of {n} (press in, then withdraw)",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    out_png = os.path.join(args.out_dir, f"figure_{tag}.png")
    cv2.imwrite(out_png, np.vstack([hdr, grid]))
    print("wrote", out_png)

    # ---- full-length video (optional) -------------------------------------
    if args.video:
        raw = os.path.join(args.out_dir, f"video_{tag}_raw.mp4")
        out_mp4 = os.path.join(args.out_dir, f"video_{tag}.mp4")
        vw = None
        for t in range(n):
            pred = preds[t]
            mask = np.linalg.norm(2 * gts[t] - 1 - FLAT, axis=-1) > 0.15
            Hs = orient_up(normal_to_height(pred, out_hw=(H_, W_)),
                           cv2.resize(mask.astype(np.uint8), (W_, H_)).astype(bool))
            cells = [cv2.cvtColor((np.clip(refs[min(t, len(refs)-1)], 0, 1)*255).astype(np.uint8),
                                  cv2.COLOR_RGB2BGR) if refs else np.zeros((CH, CW, 3), np.uint8),
                     cv2.cvtColor((np.clip(pred, 0, 1)*255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                     render_3d(Hs, os.path.join(args.out_dir, "_tmp_cloud.png")),
                     cv2.cvtColor(taxim_rgb(Hs), cv2.COLOR_RGB2BGR)]
            row = np.hstack([cv2.resize(c, (CW, CH)) for c in cells])
            if vw is None:
                vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), 5,
                                     (row.shape[1], row.shape[0]))
            vw.write(row)
            if t % 10 == 0:
                print(f"  video frame {t}/{n}", flush=True)
        vw.release()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        "-movflags", "+faststart", out_mp4], check=False)
        print("wrote", out_mp4)


if __name__ == "__main__":
    main()
