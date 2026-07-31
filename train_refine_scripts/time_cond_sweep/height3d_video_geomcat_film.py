"""Per-frame Poisson reconstruction -> 3D heightmap VIDEO for geomcat_film.

For each sim test touch, runs the model over every frame, integrates each
predicted normal map to a heightmap (Poisson), and renders a 3D relief per
frame -- stitched into an MP4 that shows the touch evolving (no-press ->
contact -> take-off). Each video frame is [predicted normal | 3D predicted |
3D ground-truth]. Sign and vertical scale are fixed ONCE per touch (from the
GT peak frame) so the bump grows smoothly instead of renormalising per frame.
"""
import os, sys, base64, subprocess
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
sys.path.insert(0, f"{ROOT}/rebot_net")
sys.path.insert(0, f"{ROOT}/baselines/RandomQuiltingTactile/TactileDreamFusion")
from dataset import TactileTransferDataset
from train import build_model
from poisson_solver import poisson_dct_neumann

TRANSFER = f"{ROOT}/log/transfer_feat_match_pseudo_mini_tactile_normal_superpoint_superglue"
COND = f"{ROOT}/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini"
CKPT = f"{ROOT}/log/rebot_checkpoints_S_geomcat_film/height3d_snapshot.pth"
VID_DIR = f"{ROOT}/log/geomcat_film_height3d_videos"
OUT = f"{ROOT}/log/tactile_normal_geomcat_film_height3d_video_report.html"
os.makedirs(VID_DIR, exist_ok=True)

TOUCHES = [(951, 2), (963, 4), (975, 1), (1000, 3)]
FLAT = np.array([0.0, 0.0, 1.0])
PANEL_H = 300            # px height of every panel
FPS = 10
VIEW = dict(elev=55, azim=-62)


def normal_to_height(rgb01):
    """RGB normal map -> plane-detrended heightmap (inverse of height->normal)."""
    n = 2.0 * rgb01 - 1.0
    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
    nz = np.clip(n[..., 2], 0.05, 1.0)
    gx, gy = -n[..., 0] / nz, -n[..., 1] / nz
    H = poisson_dct_neumann(gx, gy)
    r, c = np.mgrid[0:H.shape[0], 0:H.shape[1]]
    A = np.c_[r.ravel(), c.ravel(), np.ones(H.size)]
    coef, *_ = np.linalg.lstsq(A, H.ravel(), rcond=None)
    return H - (A @ coef).reshape(H.shape)


def surface_to_rgb(Hn):
    """Render a fixed-scale, fixed-view diffuse-lit 3D surface -> HxWx3 uint8."""
    Hs = gaussian_filter(Hn, sigma=2.6)[::3, ::3]
    rows, cols = Hs.shape
    Y, X = np.mgrid[0:rows, 0:cols]
    zy, zx = np.gradient(Hs * 9.0)
    nrm = np.dstack([-zx, -zy, np.ones_like(Hs)])
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    lgt = np.array([-0.5, -0.6, 0.7]); lgt /= np.linalg.norm(lgt)
    inten = np.clip((nrm * lgt).sum(2), 0, 1)
    shaded = cm.gray(0.28 + 0.72 * inten)
    fig = plt.figure(figsize=(3, 3), dpi=PANEL_H // 3, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1], projection="3d"); ax.set_facecolor("black")
    ax.plot_surface(X, Y, Hs, facecolors=shaded, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    ax.set_axis_off(); ax.view_init(**VIEW)
    ax.set_box_aspect((cols, rows, 0.42 * max(rows, cols)))
    ax.set_zlim(-0.3, 1.3)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return cv2.resize(buf, (PANEL_H, PANEL_H))


def label_panel(img_rgb, text):
    """Add a black title strip with white text on top of an RGB panel."""
    bar = np.zeros((26, img_rgb.shape[1], 3), np.uint8)
    out = np.vstack([bar, img_rgb])
    cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model("rebot_S", cond_chans=3, film_chans=0,
                        bottleneck_hw=24, time_cond="film").to(device)
    ck = torch.load(CKPT, map_location=device)
    model.load_state_dict(ck["model_state"]); model.eval()
    print(f"loaded geomcat_film @ epoch {ck.get('epoch','?')}")

    ds = TactileTransferDataset(TRANSFER, sorted({o for o, _ in TOUCHES}), split="test",
                                cond_dir=COND, film_modality="normal", film_scale=100,
                                geom_concat=True, video_type="tactile_normal", time_cond="film")

    made = []
    for obj, pair in TOUCHES:
        if not ds.lq_video_exists(obj, pair):
            print(f"skip {obj}_{pair}"); continue
        preds, gts = [], []
        with torch.no_grad():
            for lq, gt, blank, film, t_norm in ds.iter_video_pairs(obj, pair):
                t_in = torch.tensor([t_norm], device=device)
                pr = model(lq.unsqueeze(0).to(device), film=None, t=t_in).squeeze(0)
                preds.append(pr.cpu().clamp(0, 1).permute(1, 2, 0).numpy())
                gts.append(gt.permute(1, 2, 0).numpy())
        nfr = len(preds)
        if nfr == 0:
            continue

        Hp = [normal_to_height(p) for p in preds]
        Hg = [normal_to_height(g) for g in gts]
        dev = [np.linalg.norm(2 * g - 1 - FLAT, axis=-1).mean() for g in gts]
        pk = int(np.argmax(dev))
        mask = np.linalg.norm(2 * gts[pk] - 1 - FLAT, axis=-1) > 0.15
        # fix sign ONCE (contact raised) and z-scale ONCE (GT peak), for all frames
        sign = -1.0 if Hg[pk][mask].mean() < Hg[pk][~mask].mean() else 1.0
        S = np.percentile(np.abs(sign * Hg[pk]), 99.5) + 1e-6

        out_path = f"{VID_DIR}/{obj}_{pair}_height3d.mp4"
        raw_path = f"{VID_DIR}/{obj}_{pair}_raw.mp4"     # cv2 writes MPEG-4 Part 2
        vw = None
        for i in range(nfr):
            npanel = cv2.resize((np.clip(preds[i], 0, 1) * 255).astype(np.uint8),
                                (int(PANEL_H * 320 / 240), PANEL_H))
            p3d = surface_to_rgb(sign * Hp[i] / S)
            g3d = surface_to_rgb(sign * Hg[i] / S)
            row = np.hstack([label_panel(npanel, "Predicted normal"),
                             label_panel(p3d, "3D - predicted"),
                             label_panel(g3d, "3D - ground truth")])
            if vw is None:
                vw = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     FPS, (row.shape[1], row.shape[0]))
            vw.write(cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        vw.release()
        # Transcode to browser-playable H.264 (yuv420p, faststart). cv2's mp4v
        # writes MPEG-4 Part 2, which <video> elements cannot decode.
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw_path,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        "-movflags", "+faststart", out_path], check=True)
        os.remove(raw_path)
        made.append((obj, pair, nfr, pk, out_path))
        print(f"{obj}_{pair}: {nfr} frames, peak {pk} -> {os.path.basename(out_path)} "
              f"({os.path.getsize(out_path)/1e6:.1f}MB)")

    # ---- HTML ----
    def b64(p):
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    vids = []
    for obj, pair, nfr, pk, path in made:
        vids.append(f"""
      <section class="touch">
        <h3>Object {obj}, contact {pair} <span class="sub">{nfr} frames, peak contact ~{pk+1}</span></h3>
        <video controls autoplay loop muted playsinline>
          <source src="data:video/mp4;base64,{b64(path)}" type="video/mp4"></video>
      </section>""")

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>3D reconstruction videos (geomcat_film)</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
 :root{{color-scheme:dark;--bg:#0c0c0c;--s2:#1a1a19;--bd:#2c2c2a;--tx:#fff;--t2:#c3c2b7;--tm:#8f8d84}}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--tx);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
 .wrap{{max-width:1040px;margin:0 auto;padding:40px 24px 80px}}
 h1{{font-size:23px;margin:0 0 4px}} .meta{{color:var(--tm);font-size:13px;margin-bottom:20px}}
 .callout{{background:var(--s2);border:1px solid var(--bd);border-radius:10px;padding:16px 18px;margin:14px 0 24px;font-size:14px;color:var(--t2)}} .callout b{{color:var(--tx)}}
 .touch{{margin:26px 0;padding-top:20px;border-top:1px solid var(--bd)}}
 h3{{font-size:13px;color:var(--tm);font-weight:600;margin:0 0 12px;text-transform:uppercase;letter-spacing:.03em}}
 h3 .sub{{color:var(--t2);font-weight:400;text-transform:none;letter-spacing:0;margin-left:8px}}
 video{{width:100%;height:auto;display:block;border-radius:6px;border:1px solid var(--bd);background:#000}}
 footer{{margin-top:44px;color:var(--tm);font-size:12px}}
</style></head><body><div class="wrap">
 <h1>Watching the touch form in 3D</h1>
 <div class="meta">geomcat_film &middot; sim test set &middot; per-frame Poisson reconstruction &middot; {len(made)} touches</div>
 <div class="callout"><b>What you're seeing.</b> Every frame of each touch is passed through the network, and its
 predicted normal map is integrated to a heightmap with the Poisson solver, then rendered in 3D. Left: the predicted
 normal map (the model's raw output). Middle: its 3D reconstruction. Right: the ground-truth 3D reconstruction. The
 height sign and vertical scale are fixed once per touch (from the ground-truth peak frame), so you can watch the
 contact genuinely grow and release over the press &mdash; no-press &rarr; contact &rarr; take-off &mdash; and compare
 the predicted geometry against ground truth frame by frame.</div>
 {''.join(vids)}
 <footer>Poisson solver: baselines/.../poisson_solver.py &middot; checkpoint: log/rebot_checkpoints_S_geomcat_film &middot;
 videos: log/geomcat_film_height3d_videos/ &middot; log/tactile_normal_geomcat_film_height3d_video_report.html</footer>
</div></body></html>"""
    with open(OUT, "w") as f:
        f.write(doc)
    print("written:", OUT, "| videos:", len(made))


if __name__ == "__main__":
    main()
