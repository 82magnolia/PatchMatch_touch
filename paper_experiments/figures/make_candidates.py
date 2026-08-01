"""Render 4-5 candidate examples for each figure in paper_source/figures/.

Selection is driven by log/paper_figure_candidates/descriptors.json (see
score_candidates.py). Each figure ranks touches by its OWN objective, because
what makes a good teaser is not what makes a good ablation panel:

  teaser        structure + the reference and query poses must actually differ
                (ref_query_dist), else the "analogy" is a near-copy + good result
  gt_retrieval  structure + gain, so the refinement step is visibly doing work
  ablation      structure + abl_gap, so removing a component visibly degrades
  recon         relief + contact area, so there is 3D shape to integrate
  full_pipeline structure + our margin over the baselines (scored on job 2)

Candidates are z-scored per objective and de-duplicated across objects so the
5 options are genuinely different scenes rather than neighbouring touches.

3D panels reuse train_refine_scripts/time_cond_sweep/height3d_geomcat_film.py and
the RGB visuo-tactile panels reuse
train_refine_scripts/predicted_normal_to_rgb/predicted_normal_to_rgb.py, so the
figures match the pipeline those scripts define.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
sys.path.insert(0, os.path.join(ROOT, "rebot_net"))
sys.path.insert(0, os.path.join(ROOT, "train_refine_scripts/time_cond_sweep"))
sys.path.insert(0, os.path.join(ROOT, "train_refine_scripts/predicted_normal_to_rgb"))
sys.path.insert(0, os.path.join(ROOT, "baselines/RandomQuiltingTactile/TactileDreamFusion"))

from dataset import TactileTransferDataset          # noqa: E402
from train import build_model                       # noqa: E402

DESC = os.path.join(ROOT, "log/paper_figure_candidates/descriptors.json")
OUT_ROOT = os.path.join(ROOT, "log/paper_figure_candidates")
J1_TRANSFER = os.path.join(ROOT, "log/paper_job1_transfer_normal")
J1_COND = os.path.join(ROOT, "Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini")
REF_BASE = os.path.join(ROOT, "Taxim/results/gen_contact_full_tactile_normal_pseudo_mini")
CKPTS = {
    "ours": (os.path.join(ROOT, "log/rebot_checkpoints_S_geomcat_film/best.pth"), "film", True),
    "wo_film": (os.path.join(ROOT, "log/rebot_checkpoints_S_geomcat_none/best.pth"), "none", True),
    "wo_cat": (os.path.join(ROOT,
               "log/rebot_checkpoints_S_pseudo_mini_tactile_normal_superpoint_superglue_cond-film-normal/best.pth"),
               "none", False),
}
FLAT = np.array([0.0, 0.0, 1.0])
W, H = 320, 240


def z(v):
    v = np.asarray(v, float)
    m, s = np.nanmean(v), np.nanstd(v)
    return (v - m) / (s if s > 1e-9 else 1.0)


def load_desc():
    with open(DESC) as f:
        return json.load(f)


def pick(rows, score, n=5, per_object=1):
    order = np.argsort(-score)
    out, seen = [], {}
    for i in order:
        r = rows[i]
        if seen.get(r["object"], 0) >= per_object:
            continue
        seen[r["object"]] = seen.get(r["object"], 0) + 1
        out.append((r, float(score[i])))
        if len(out) == n:
            break
    return out


# ---------------------------------------------------------------- inference

_models = {}


def get_model(arm, device):
    if arm in _models:
        return _models[arm]
    ckpt, time_cond, geom = CKPTS[arm]
    cond_chans = 3 if geom else 0
    film_chans = 0 if geom else 3
    m = build_model("rebot_S", cond_chans=cond_chans, film_chans=film_chans,
                    bottleneck_hw=24, time_cond=time_cond).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    m.eval()
    _models[arm] = (m, geom)
    return _models[arm]


def predict(arm, obj, touch, device):
    """Return (pred_frames, gt_frames, lq_frames) for one touch."""
    _, geom = get_model(arm, device)
    ds = TactileTransferDataset(
        J1_TRANSFER, [obj], split="test", cond_dir=J1_COND,
        film_modality="normal", film_scale=100, geom_concat=geom,
        video_type="tactile_normal", time_cond=CKPTS[arm][1])
    model, _ = _models[arm]
    preds, gts, lqs = [], [], []
    with torch.no_grad():
        for lq, gt, _blank, film, t_norm in ds.iter_video_pairs(obj, touch):
            t_in = torch.tensor([t_norm], device=device)
            film_in = film.unsqueeze(0).to(device) if film is not None else None
            pr = model(lq.unsqueeze(0).to(device), film=film_in, t=t_in).squeeze(0)
            preds.append(pr.cpu().clamp(0, 1).permute(1, 2, 0).numpy())
            gts.append(gt.permute(1, 2, 0).numpy())
            lqs.append(lq[1, :3].permute(1, 2, 0).numpy())
    return preds, gts, lqs


# ---------------------------------------------------------------- drawing

def cell(img, caption):
    if img is None:
        img = np.full((H, W, 3), 245, np.uint8)
        cv2.putText(img, "n/a", (135, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (130, 130, 130), 1, cv2.LINE_AA)
    else:
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img = cv2.resize(img, (W, H))
    strip = np.full((26, W, 3), 255, np.uint8)
    cv2.putText(strip, caption[:44], (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([strip, img])


def save(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def read_jpg(p):
    im = cv2.imread(p)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im is not None else None


def read_video(p):
    if not os.path.exists(p):
        return None
    c = cv2.VideoCapture(p)
    fs = []
    while True:
        ok, f = c.read()
        if not ok:
            break
        fs.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    c.release()
    return fs or None


# ---------------------------------------------------------------- figures

def fig_teaser(rows, device, n=5):
    """A:A' :: B:B'  -- reference geometry/touch over query geometry/prediction."""
    score = (z([r["structure"] for r in rows])
             + z([r["ref_query_dist"] for r in rows])
             + 0.5 * z([r["psnr_ours"] for r in rows]))
    out, cands = os.path.join(OUT_ROOT, "teaser"), []
    for r, s in pick(rows, score, n):
        o, t = r["object"], r["touch"]
        preds, gts, _ = predict("ours", o, t, device)
        if not preds:
            continue
        k = r["peak_frame"]
        ref_v = read_video(os.path.join(J1_TRANSFER, str(o), f"{t}_ref_tactile_normal.mp4"))
        panels = {
            "A_ref_normal": read_jpg(os.path.join(REF_BASE, str(o), f"{t}_scale100_normal.jpg")),
            "Ap_ref_touch": ref_v[min(k, len(ref_v) - 1)] if ref_v else None,
            "B_query_normal": read_jpg(os.path.join(J1_COND, str(o), f"{t}_scale100_normal.jpg")),
            "Bp_pred_touch": preds[min(k, len(preds) - 1)],
        }
        for nm, im in panels.items():
            save(cell(im, nm), os.path.join(out, f"{o}_{t}", f"{nm}.png"))
        grid = np.vstack([
            np.hstack([cell(panels["A_ref_normal"], "A: reference geometry"),
                       cell(panels["Ap_ref_touch"], "A': reference touch")]),
            np.hstack([cell(panels["B_query_normal"], "B: query geometry"),
                       cell(panels["Bp_pred_touch"], "B': our prediction")])])
        save(grid, os.path.join(out, f"{o}_{t}", "preview.png"))
        cands.append({"object": o, "touch": t, "score": s,
                      "structure": r["structure"], "ref_query_dist": r["ref_query_dist"],
                      "psnr_ours": r["psnr_ours"],
                      "preview": f"log/paper_figure_candidates/teaser/{o}_{t}/preview.png"})
        print(f"  teaser {o}_{t}  score {s:.2f}")
    return cands


def fig_gt_retrieval(rows, device, n=5):
    """3 x 6 matrix rows -- one candidate row per touch."""
    score = (z([r["structure"] for r in rows]) + z([r["gain"] for r in rows])
             + 0.5 * z([r["psnr_ours"] for r in rows]))
    out, cands = os.path.join(OUT_ROOT, "gt_retrieval"), []
    for r, s in pick(rows, score, n):
        o, t = r["object"], r["touch"]
        preds, gts, lqs = predict("ours", o, t, device)
        if not preds:
            continue
        k = min(r["peak_frame"], len(preds) - 1)
        ref_v = read_video(os.path.join(J1_TRANSFER, str(o), f"{t}_ref_tactile_normal.mp4"))
        cols = [
            ("01_ref_touch", ref_v[min(k, len(ref_v) - 1)] if ref_v else None, "reference touch"),
            ("02_ref_normal", read_jpg(os.path.join(REF_BASE, str(o), f"{t}_scale100_normal.jpg")), "reference normal"),
            ("03_query_normal", read_jpg(os.path.join(J1_COND, str(o), f"{t}_scale100_normal.jpg")), "query normal"),
            ("04_coarse", lqs[k], "coarse transfer"),
            ("05_refined", preds[k], "refined (ours)"),
            ("06_gt_query", gts[k], "ground truth"),
        ]
        strip = []
        for nm, im, cap in cols:
            save(cell(im, cap), os.path.join(out, f"{o}_{t}", f"{nm}.png"))
            strip.append(cell(im, cap))
        save(np.hstack(strip), os.path.join(out, f"{o}_{t}", "preview.png"))
        cands.append({"object": o, "touch": t, "score": s, "structure": r["structure"],
                      "gain": r["gain"], "psnr_ours": r["psnr_ours"],
                      "preview": f"log/paper_figure_candidates/gt_retrieval/{o}_{t}/preview.png"})
        print(f"  gt_retrieval {o}_{t}  score {s:.2f}  gain {r['gain']:.1f} dB")
    return cands


def fig_ablation(rows, device, n=5):
    """coarse | w/o FiLM | w/o normal concat | ours | GT."""
    score = (z([r["abl_gap"] for r in rows]) + 0.7 * z([r["structure"] for r in rows]))
    out, cands = os.path.join(OUT_ROOT, "ablation"), []
    for r, s in pick(rows, score, n):
        o, t = r["object"], r["touch"]
        preds, gts, lqs = predict("ours", o, t, device)
        if not preds:
            continue
        k = min(r["peak_frame"], len(preds) - 1)
        wof, _, _ = predict("wo_film", o, t, device)
        woc, _, _ = predict("wo_cat", o, t, device)
        cols = [("1_no_refine", lqs[k], "w/o refinement"),
                ("2_wo_film", wof[k] if wof else None, "w/o temporal FiLM"),
                ("3_wo_cat", woc[k] if woc else None, "w/o normal concat"),
                ("4_ours", preds[k], "ours (full)"),
                ("5_gt", gts[k], "ground truth")]
        strip = []
        for nm, im, cap in cols:
            save(cell(im, cap), os.path.join(out, f"{o}_{t}", f"{nm}.png"))
            strip.append(cell(im, cap))
        save(np.hstack(strip), os.path.join(out, f"{o}_{t}", "preview.png"))
        cands.append({"object": o, "touch": t, "score": s, "abl_gap": r["abl_gap"],
                      "structure": r["structure"], "psnr_ours": r["psnr_ours"],
                      "psnr_wo_film": r["psnr_wo_film"], "psnr_wo_cat": r["psnr_wo_cat"],
                      "preview": f"log/paper_figure_candidates/ablation/{o}_{t}/preview.png"})
        print(f"  ablation {o}_{t}  gap {r['abl_gap']:.1f} dB")
    return cands


def fig_recon(rows, device, n=5, n_cols=4):
    """ref video | predicted video | 3D relief | simulated visuo-tactile RGB."""
    from height3d_geomcat_film import normal_to_height, orient_up, render_3d
    from predicted_normal_to_rgb import normal_to_taxim

    score = (z([r["relief"] for r in rows]) + z([r["contact_frac"] for r in rows])
             + 0.5 * z([r["psnr_ours"] for r in rows]))
    out, cands = os.path.join(OUT_ROOT, "recon"), []
    for r, s in pick(rows, score, n):
        o, t = r["object"], r["touch"]
        preds, gts, _ = predict("ours", o, t, device)
        if not preds:
            continue
        ref_v = read_video(os.path.join(J1_TRANSFER, str(o), f"{t}_ref_tactile_normal.mp4"))
        # sample columns from the in-contact portion only
        dev = np.array([np.linalg.norm(2 * g - 1 - FLAT, axis=-1).mean() for g in gts])
        idx = np.where(dev > dev.max() * 0.45)[0]
        if len(idx) < n_cols:
            idx = np.arange(len(gts))
        cols_i = np.linspace(idx[0], idx[-1], n_cols).round().astype(int)

        d = os.path.join(out, f"{o}_{t}")
        rows_img = [[], [], [], []]
        for ci, fi in enumerate(cols_i):
            rp = ref_v[min(fi, len(ref_v) - 1)] if ref_v else None
            pr = preds[fi]
            Hh = orient_up(normal_to_height(pr),
                           np.linalg.norm(2 * gts[fi] - 1 - FLAT, axis=-1) > 0.15)
            p3d = os.path.join(d, f"c{ci}_3d.png")
            os.makedirs(d, exist_ok=True)
            render_3d(Hh, p3d)
            rgb = normal_to_taxim(pr)
            rows_img[0].append(cell(rp, f"ref f{fi}"))
            rows_img[1].append(cell(pr, f"pred f{fi}"))
            rows_img[2].append(cell(read_jpg(p3d), f"3D relief f{fi}"))
            rows_img[3].append(cell(rgb, f"visuo-tactile RGB f{fi}"))
            save(cell(pr, f"pred f{fi}"), os.path.join(d, f"c{ci}_pred.png"))
            save(cell(rgb, f"rgb f{fi}"), os.path.join(d, f"c{ci}_rgb.png"))
        save(np.vstack([np.hstack(r_) for r_ in rows_img]),
             os.path.join(d, "preview.png"))
        cands.append({"object": o, "touch": t, "score": s, "relief": r["relief"],
                      "contact_frac": r["contact_frac"], "psnr_ours": r["psnr_ours"],
                      "preview": f"log/paper_figure_candidates/recon/{o}_{t}/preview.png"})
        print(f"  recon {o}_{t}  relief {r['relief']:.0f}")
    return cands


BUILDERS = {"teaser": fig_teaser, "gt_retrieval": fig_gt_retrieval,
            "ablation": fig_ablation, "recon": fig_recon}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", nargs="+", default=list(BUILDERS), choices=list(BUILDERS))
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_desc()
    all_c = {}
    for fig in args.figures:
        print(f"\n=== {fig} ===")
        all_c[fig] = BUILDERS[fig](rows, device, args.n)
    path = os.path.join(OUT_ROOT, "candidates.json")
    existing = json.load(open(path)) if os.path.exists(path) else {}
    existing.update(all_c)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
