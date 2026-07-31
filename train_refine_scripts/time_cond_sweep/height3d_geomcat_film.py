"""Predicted-normal -> heightmap (Poisson) -> 3D relief report for geomcat_film.

Runs the geom-concat + sinusoidal-FiLM model on sim test touches, decodes each
predicted tactile-normal map, integrates it to a heightmap with the codebase's
Poisson DCT/Neumann solver, and renders a hillshaded 3D surface (like the
'Sensor | 3D Reconstruction' reference). Ground-truth 3D shown alongside.
"""
import os, sys, io, base64, html
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
sys.path.insert(0, f"{ROOT}/rebot_net")
sys.path.insert(0, f"{ROOT}/baselines/RandomQuiltingTactile/TactileDreamFusion")
from dataset import TactileTransferDataset
from train import build_model
from poisson_solver import poisson_dct_neumann

TRANSFER = f"{ROOT}/log/transfer_feat_match_pseudo_mini_tactile_normal_superpoint_superglue"
COND = f"{ROOT}/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini"
CKPT = f"{ROOT}/log/rebot_checkpoints_S_geomcat_film/height3d_snapshot.pth"
FRAME_DIR = f"{ROOT}/log/geomcat_film_height3d_frames"
OUT = f"{ROOT}/log/tactile_normal_geomcat_film_height3d_report.html"
os.makedirs(FRAME_DIR, exist_ok=True)

TOUCHES = [(951, 2), (963, 4), (975, 1), (988, 6), (1000, 3), (955, 0), (992, 5), (970, 2)]
FLAT = np.array([0.0, 0.0, 1.0])


def decode_normal(rgb01):
    """(H,W,3) RGB in [0,1] -> unit surface normals (nx,ny,nz), nz>0."""
    n = 2.0 * rgb01 - 1.0
    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
    nz = np.clip(n[..., 2], 0.05, 1.0)
    return n[..., 0], n[..., 1], nz


def normal_to_height(rgb01):
    """RGB normal map -> detrended heightmap, contact oriented upward.

    Inverts simOptical.height_map_to_normals: normal=[-dzdx,-dzdy,1]/denom, so
    gx=dz/dx=-nx/nz, gy=dz/dy=-ny/nz; then Poisson-integrate.
    """
    nx, ny, nz = decode_normal(rgb01)
    gx, gy = -nx / nz, -ny / nz
    H = poisson_dct_neumann(gx, gy)
    # remove global tilt (integration is only defined up to a plane): least-sq
    # plane fit over the whole frame, subtracted.
    r, c = np.mgrid[0:H.shape[0], 0:H.shape[1]]
    A = np.c_[r.ravel(), c.ravel(), np.ones(H.size)]
    coef, *_ = np.linalg.lstsq(A, H.ravel(), rcond=None)
    H = H - (A @ coef).reshape(H.shape)
    return H


def orient_up(H, contact_mask):
    """Flip sign so the contact region is raised relative to the background."""
    if contact_mask.sum() < 10:
        return H
    if H[contact_mask].mean() < H[~contact_mask].mean():
        H = -H
    return H


def render_3d(H, out_path):
    from scipy.ndimage import gaussian_filter
    Hn = H - H.min()
    Hn = Hn / (Hn.max() + 1e-8)
    Hn = gaussian_filter(Hn, sigma=2.6)      # smooth the integration stepping
    ds = 2                                    # lighter mesh
    Hn = Hn[::ds, ::ds]
    rows, cols = Hn.shape
    Y, X = np.mgrid[0:rows, 0:cols]
    # Direct Lambertian (diffuse) shading from the surface normals: smooth matte
    # relief, avoiding the contour rings that LightSource.shade's soft-blend
    # hillshade produces on these lightly-banded (video-compressed) normals.
    zy, zx = np.gradient(Hn * 9.0)
    nrm = np.dstack([-zx, -zy, np.ones_like(Hn)])
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    lgt = np.array([-0.5, -0.6, 0.7]); lgt /= np.linalg.norm(lgt)
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


def save_normal_png(rgb01, out_path):
    fig = plt.figure(figsize=(4.2, 3.2), facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.imshow(np.clip(rgb01, 0, 1))
    fig.savefig(out_path, dpi=110, facecolor="black")
    plt.close(fig)


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model("rebot_S", cond_chans=3, film_chans=0,
                        bottleneck_hw=24, time_cond="film").to(device)
    ck = torch.load(CKPT, map_location=device)
    model.load_state_dict(ck["model_state"]); model.eval()
    print(f"loaded geomcat_film @ epoch {ck.get('epoch','?')}")

    ds = TactileTransferDataset(TRANSFER, sorted({o for o, _ in TOUCHES}), split="test",
                                cond_dir=COND, film_modality="normal", film_scale=100,
                                geom_concat=True, video_type="tactile_normal",
                                time_cond="film")

    cards = []
    for obj, pair in TOUCHES:
        if not ds.lq_video_exists(obj, pair):
            print(f"skip {obj}_{pair} (missing)"); continue
        preds, gts = [], []
        with torch.no_grad():
            for lq, gt, blank, film, t_norm in ds.iter_video_pairs(obj, pair):
                t_in = torch.tensor([t_norm], device=device)
                pr = model(lq.unsqueeze(0).to(device), film=None, t=t_in).squeeze(0)
                preds.append(pr.cpu().clamp(0, 1).permute(1, 2, 0).numpy())
                gts.append(gt.permute(1, 2, 0).numpy())
        if not preds:
            continue
        # peak-contact frame: max mean deviation of the GT normal from flat
        dev = [np.linalg.norm(2 * g - 1 - FLAT, axis=-1).mean() for g in gts]
        k = int(np.argmax(dev))
        pred_rgb, gt_rgb = preds[k], gts[k]
        mask = np.linalg.norm(2 * gt_rgb - 1 - FLAT, axis=-1) > 0.15   # contact region

        Hp = orient_up(normal_to_height(pred_rgb), mask)
        Hg = orient_up(normal_to_height(gt_rgb), mask)

        base = f"{FRAME_DIR}/{obj}_{pair}"
        save_normal_png(pred_rgb, f"{base}_prednormal.png")
        save_normal_png(gt_rgb, f"{base}_gtnormal.png")
        render_3d(Hp, f"{base}_pred3d.png")
        render_3d(Hg, f"{base}_gt3d.png")
        cards.append((obj, pair, k, len(preds), base))
        print(f"{obj}_{pair}: peak frame {k}/{len(preds)} done")

    # ---- HTML ----
    def img(p): return f'<img src="data:image/png;base64,{b64(p)}" alt="">'
    secs = []
    for obj, pair, k, nfr, base in cards:
        secs.append(f"""
      <section class="touch">
        <h3>Object {obj}, contact {pair} <span class="sub">peak-contact frame {k+1}/{nfr}</span></h3>
        <div class="grid">
          <figure>{img(base+'_prednormal.png')}<figcaption>Predicted normal map (model output)</figcaption></figure>
          <figure>{img(base+'_pred3d.png')}<figcaption>3D reconstruction &mdash; predicted</figcaption></figure>
          <figure>{img(base+'_gt3d.png')}<figcaption>3D reconstruction &mdash; ground truth</figcaption></figure>
        </div>
      </section>""")

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Predicted normals &rarr; 3D heightmaps (geomcat_film)</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
 :root{{color-scheme:dark;--bg:#0c0c0c;--s2:#1a1a19;--bd:#2c2c2a;--tx:#fff;--t2:#c3c2b7;--tm:#8f8d84}}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--tx);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
 .wrap{{max-width:900px;margin:0 auto;padding:40px 24px 80px}}
 h1{{font-size:23px;margin:0 0 4px}} .meta{{color:var(--tm);font-size:13px;margin-bottom:20px}}
 .callout{{background:var(--s2);border:1px solid var(--bd);border-radius:10px;padding:16px 18px;margin:14px 0 24px;font-size:14px;color:var(--t2)}} .callout b{{color:var(--tx)}}
 .touch{{margin:26px 0;padding-top:20px;border-top:1px solid var(--bd)}}
 h3{{font-size:13px;color:var(--tm);font-weight:600;margin:0 0 12px;text-transform:uppercase;letter-spacing:.03em}}
 h3 .sub{{color:var(--t2);font-weight:400;text-transform:none;letter-spacing:0;margin-left:8px}}
 .grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}}
 @media(max-width:680px){{.grid{{grid-template-columns:1fr}}}}
 figure{{margin:0}} img{{width:100%;height:auto;display:block;border-radius:6px;border:1px solid var(--bd);background:#000}}
 figcaption{{font-size:11.5px;color:var(--t2);margin-top:5px;text-align:center}}
 footer{{margin-top:44px;color:var(--tm);font-size:12px}}
</style></head><body><div class="wrap">
 <h1>From predicted normals to 3D touch geometry</h1>
 <div class="meta">geomcat_film (normals concatenated + sinusoidal-FiLM time) &middot; sim test set &middot; {len(cards)} touches</div>
 <div class="callout"><b>Pipeline.</b> The network predicts a per-frame surface-normal map in the tactile-normal domain.
 We decode it to unit normals, convert to surface gradients (g<sub>x</sub>=&minus;n<sub>x</sub>/n<sub>z</sub>,
 g<sub>y</sub>=&minus;n<sub>y</sub>/n<sub>z</sub>, inverting the simulator's height&rarr;normal map), and integrate them
 into a heightmap with the codebase's Poisson DCT/Neumann solver. The heightmap is then rendered as a hillshaded 3D
 surface. For each touch we show the peak-contact frame: the predicted normal map, its 3D reconstruction, and the
 ground-truth 3D reconstruction for comparison.</div>
 {''.join(secs)}
 <footer>Poisson solver: baselines/RandomQuiltingTactile/TactileDreamFusion/poisson_solver.py &middot;
 checkpoint: log/rebot_checkpoints_S_geomcat_film &middot; log/tactile_normal_geomcat_film_height3d_report.html</footer>
</div></body></html>"""
    with open(OUT, "w") as f:
        f.write(doc)
    print("written:", OUT, "| touches:", len(cards))


if __name__ == "__main__":
    main()
