"""PREDICTED NORMAL MAP -> RGB TACTILE IMAGE (Taxim optical simulation).

This is the canonical "predicted normal -> RGB" converter. The core is
`normal_to_taxim(normal_map)`: it integrates a predicted (or GT) tactile-normal
map to a heightmap (Poisson), then applies Taxim's optical simulation
(`taxim_rgb`: calibrated gradient->RGB lookup table + gel background) to
synthesize the GelSight RGB tactile image.

As driven by `main()` it runs geomcat_film over sim test touches and stitches an
H.264 MP4 per touch -- [predicted normal | Taxim tactile (predicted) | Taxim
tactile (ground truth)] -- plus an HTML report. To convert your own normal maps,
import `normal_to_taxim` / `taxim_rgb` directly.
"""
import os, sys, base64, subprocess
import numpy as np, cv2, torch
from scipy.ndimage import gaussian_filter

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
sys.path.insert(0, f"{ROOT}/rebot_net")
sys.path.insert(0, f"{ROOT}/Taxim")
sys.path.insert(0, f"{ROOT}/baselines/RandomQuiltingTactile/TactileDreamFusion")
from dataset import TactileTransferDataset
from train import build_model
from poisson_solver import poisson_dct_neumann
from Basics.CalibData import CalibData
import Basics.sensorParams as psp
import Basics.params as pr

TRANSFER = f"{ROOT}/log/transfer_feat_match_pseudo_mini_tactile_normal_superpoint_superglue"
COND = f"{ROOT}/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini"
CKPT = f"{ROOT}/log/rebot_checkpoints_S_geomcat_film/height3d_snapshot.pth"
CAL = f"{ROOT}/Taxim/calibs"
VID_DIR = f"{ROOT}/log/geomcat_film_taxim_rgb_videos"
OUT = f"{ROOT}/log/tactile_normal_geomcat_film_taxim_rgb_report.html"
os.makedirs(VID_DIR, exist_ok=True)

TOUCHES = [(951, 2), (963, 4), (975, 1), (1000, 3)]
H_, W_ = psp.h, psp.w                      # sensor resolution (480x640)
PANEL_H = 300
FPS = 10

# --- Taxim optical simulation (no shadow), precomputed constants ---
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
    gd = np.zeros_like(mag); v = mag != 0
    gd[v] = np.arctan2(dzdx[v] / mag[v], dzdy[v] / mag[v])
    return np.pad(gm, 1, mode="edge"), np.pad(gd, 1, mode="edge")


def taxim_rgb(H):
    """Heightmap (H_,W_) -> Taxim GelSight RGB uint8 (H_,W_,3)."""
    gm, gd = _gen_normals(H)
    ix = np.clip(np.floor(gm / (0.5*np.pi/(_bins-1))).astype(int), 0, _bins-1)
    iy = np.clip(np.floor((gd+np.pi) / (2*np.pi/(_bins-1))).astype(int), 0, _bins-1)
    est = np.zeros((H_, W_, 3))
    for c, g in enumerate([_calib.grad_r, _calib.grad_g, _calib.grad_b]):
        pm = g[ix, iy, :].reshape(H_*W_, g.shape[2])
        est[:, :, c] = np.sum(_A * pm, axis=1).reshape(H_, W_)
    return np.clip(est + _BG, 0, 255).astype(np.uint8)


def normal_to_taxim(rgb01_small):
    """Predicted/GT normal map (h,w,3 in [0,1]) -> Taxim RGB, via sensor-res
    Poisson integration (resize the *normals* so slopes keep their magnitude)."""
    big = cv2.resize(rgb01_small, (W_, H_))
    n = 2.0 * big - 1.0
    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
    nz = np.clip(n[..., 2], 0.05, 1.0)
    H = poisson_dct_neumann(-n[..., 0] / nz, -n[..., 1] / nz)
    return taxim_rgb(H)


def label(img_rgb, text):
    bar = np.zeros((26, img_rgb.shape[1], 3), np.uint8)
    out = np.vstack([bar, img_rgb])
    cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def panel(img_rgb):
    return cv2.resize(img_rgb, (int(PANEL_H * 4/3), PANEL_H))   # 4:3 -> 400x300


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model("rebot_S", cond_chans=3, film_chans=0,
                        bottleneck_hw=24, time_cond="film").to(device)
    ck = torch.load(CKPT, map_location=device)
    model.load_state_dict(ck["model_state"]); model.eval()
    print(f"loaded geomcat_film @ epoch {ck.get('epoch','?')}; sensor {H_}x{W_}")

    ds = TactileTransferDataset(TRANSFER, sorted({o for o, _ in TOUCHES}), split="test",
                                cond_dir=COND, film_modality="normal", film_scale=100,
                                geom_concat=True, video_type="tactile_normal", time_cond="film")

    made = []
    for obj, pair in TOUCHES:
        if not ds.lq_video_exists(obj, pair):
            print(f"skip {obj}_{pair}"); continue
        raw = f"{VID_DIR}/{obj}_{pair}_raw.mp4"
        out_path = f"{VID_DIR}/{obj}_{pair}_taxim.mp4"
        vw = None; nfr = 0
        with torch.no_grad():
            for lq, gt, blank, film, t_norm in ds.iter_video_pairs(obj, pair):
                t_in = torch.tensor([t_norm], device=device)
                pr_rgb = model(lq.unsqueeze(0).to(device), film=None, t=t_in).squeeze(0)
                pred = pr_rgb.cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                gtn = gt.permute(1, 2, 0).numpy()
                npanel = panel((np.clip(pred, 0, 1) * 255).astype(np.uint8))
                tp = panel(normal_to_taxim(pred))
                tg = panel(normal_to_taxim(gtn))
                row = np.hstack([label(npanel, "Predicted normal"),
                                 label(tp, "Taxim tactile - predicted"),
                                 label(tg, "Taxim tactile - ground truth")])
                if vw is None:
                    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"),
                                         FPS, (row.shape[1], row.shape[0]))
                vw.write(cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
                nfr += 1
        vw.release()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        "-movflags", "+faststart", out_path], check=True)
        os.remove(raw)
        made.append((obj, pair, nfr, out_path))
        print(f"{obj}_{pair}: {nfr} frames -> {os.path.basename(out_path)} "
              f"({os.path.getsize(out_path)/1e6:.1f}MB)")

    def b64(p):
        with open(p, "rb") as f: return base64.b64encode(f.read()).decode("ascii")
    vids = "".join(f"""
      <section class="touch"><h3>Object {o}, contact {p} <span class="sub">{n} frames</span></h3>
      <video controls autoplay loop muted playsinline preload="auto">
        <source src="data:video/mp4;base64,{b64(pt)}" type="video/mp4"></video></section>"""
                   for o, p, n, pt in made)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Taxim RGB tactile from reconstructed heightmaps</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
 :root{{color-scheme:dark;--bg:#0c0c0c;--s2:#1a1a19;--bd:#2c2c2a;--tx:#fff;--t2:#c3c2b7;--tm:#8f8d84}}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--tx);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
 .wrap{{max-width:1060px;margin:0 auto;padding:40px 24px 80px}}
 h1{{font-size:23px;margin:0 0 4px}} .meta{{color:var(--tm);font-size:13px;margin-bottom:20px}}
 .callout{{background:var(--s2);border:1px solid var(--bd);border-radius:10px;padding:16px 18px;margin:14px 0 24px;font-size:14px;color:var(--t2)}} .callout b{{color:var(--tx)}}
 .touch{{margin:26px 0;padding-top:20px;border-top:1px solid var(--bd)}}
 h3{{font-size:13px;color:var(--tm);font-weight:600;margin:0 0 12px;text-transform:uppercase;letter-spacing:.03em}}
 h3 .sub{{color:var(--t2);font-weight:400;text-transform:none;letter-spacing:0;margin-left:8px}}
 video{{width:100%;height:auto;display:block;border-radius:6px;border:1px solid var(--bd);background:#000}}
 footer{{margin-top:44px;color:var(--tm);font-size:12px}}
</style></head><body><div class="wrap">
 <h1>Synthesizing the GelSight tactile image from reconstructed geometry</h1>
 <div class="meta">geomcat_film &middot; sim test set &middot; Poisson heightmap &rarr; Taxim optical simulation &middot; {len(made)} touches &middot; H.264</div>
 <div class="callout"><b>Pipeline.</b> The network predicts a normal map per frame; we integrate it to a heightmap
 (Poisson), then run <b>Taxim's optical simulation</b> &mdash; the same calibrated gradient&rarr;RGB lookup table and gel
 background used to generate the training data &mdash; to synthesize the GelSight tactile RGB image. Left: the predicted
 normal map. Middle: the Taxim tactile image rendered from the predicted geometry. Right: the Taxim tactile image
 rendered from the ground-truth normals, for comparison. The coloured fringes are the sensor's red/green/blue LED
 response to the surface slope &mdash; the hallmark of a GelSight reading.</div>
 {vids}
 <footer>Taxim calibs: Taxim/calibs/ (polycalib.npz, dataPack.npz) &middot; sensor {H_}x{W_} &middot;
 videos: log/geomcat_film_taxim_rgb_videos/ &middot; log/tactile_normal_geomcat_film_taxim_rgb_report.html</footer>
</div></body></html>"""
    with open(OUT, "w") as f:
        f.write(doc)
    print("written:", OUT, "| videos:", len(made))


if __name__ == "__main__":
    main()
