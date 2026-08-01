"""Score every evaluation touch so figure candidates can be chosen on evidence.

Picking qualitative examples by eye (or by PSNR alone) is a trap: the
highest-PSNR touches are the flattest, most featureless contacts, which are easy
to predict and show the reader nothing. So for each touch we compute both how
much there is to SEE and how well the method DID, then each figure ranks with its
own objective.

Descriptors, per (object, touch) of the ground-truth-retrieval eval set:

  structure     std of the GT peak-contact frame -- how much surface detail is visible
  contact_frac  fraction of pixels whose normal deviates from flat, i.e. contact area
  relief        peak-to-peak of the Poisson-integrated GT heightmap -- 3D relief depth
  psnr_coarse   our coarse transfer quality
  psnr_ours     our refined quality
  gain          psnr_ours - psnr_coarse: how visibly the refinement stage helps
  abl_gap       psnr_ours - max(psnr_wo_film, psnr_wo_cat): how visibly the
                conditioning ablations degrade, i.e. how much an ablation figure
                drawn from this touch would actually show
  ref_query_dist mean abs difference between the reference and query normal
                renders -- how different the two poses are, so a teaser drawn
                from this touch is a real analogy rather than a near-copy

Writes log/paper_figure_candidates/descriptors.json.
"""
import json
import os
import pickle
import sys

import cv2
import numpy as np

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
sys.path.insert(0, os.path.join(ROOT, "train_refine_scripts/time_cond_sweep"))
sys.path.insert(0, os.path.join(ROOT, "baselines/RandomQuiltingTactile/TactileDreamFusion"))

TRANSFER = os.path.join(ROOT, "log/paper_job1_transfer_normal")
REF_BASE = os.path.join(ROOT, "Taxim/results/gen_contact_full_tactile_normal_pseudo_mini")
QUERY_BASE = os.path.join(ROOT, "Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini")
OUT_DIR = os.path.join(ROOT, "log/paper_figure_candidates")
FLAT = np.array([0.0, 0.0, 1.0])


def load_json(p):
    p = os.path.join(ROOT, p)
    return json.load(open(p)) if os.path.exists(p) else {}


def read_video(path):
    if not os.path.exists(path):
        return None
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    cap.release()
    return frames or None


def peak_frame(frames):
    """Index of deepest contact: max mean deviation of the normal from flat."""
    dev = [np.linalg.norm(2 * f - 1 - FLAT, axis=-1).mean() for f in frames]
    return int(np.argmax(dev)), float(np.max(dev))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from height3d_geomcat_film import normal_to_height   # reuse the paper's solver path

    ours = load_json("log/paper_job1_refine_ours_normal/metrics.json").get("per_touch", {})
    wof = load_json("log/paper_job1_refine_wo_film_normal/metrics.json").get("per_touch", {})
    woc = load_json("log/paper_job1_refine_wo_normalcat_normal/metrics.json").get("per_touch", {})

    rows = []
    objs = sorted(int(d) for d in os.listdir(TRANSFER) if d.isdigit())
    for oi, obj in enumerate(objs):
        cp = os.path.join(TRANSFER, str(obj), "metrics.pkl")
        coarse = pickle.load(open(cp, "rb"))["per_touch"] if os.path.exists(cp) else {}
        for t in range(8):
            key = f"{obj}_{t}"
            if key not in ours:
                continue
            gt = read_video(os.path.join(TRANSFER, str(obj), f"{t}_query_tactile_normal.mp4"))
            if gt is None:
                continue
            k, dev = peak_frame(gt)
            g = gt[k]

            mask = np.linalg.norm(2 * g - 1 - FLAT, axis=-1) > 0.15
            H = normal_to_height(g)
            relief = float(np.percentile(H, 99) - np.percentile(H, 1))

            rn = cv2.imread(os.path.join(REF_BASE, str(obj), f"{t}_scale100_normal.jpg"))
            qn = cv2.imread(os.path.join(QUERY_BASE, str(obj), f"{t}_scale100_normal.jpg"))
            if rn is None or qn is None:
                rq = 0.0
            else:
                rq = float(np.abs(rn.astype(np.float32) - qn.astype(np.float32)).mean())

            p_ours = ours[key]["PSNR"]
            p_coarse = float(coarse.get(t, {}).get("PSNR", np.nan))
            p_wof = wof.get(key, {}).get("PSNR", np.nan)
            p_woc = woc.get(key, {}).get("PSNR", np.nan)

            rows.append({
                "object": obj, "touch": t, "peak_frame": k, "n_frames": len(gt),
                "structure": float(g.std()),
                "contact_frac": float(mask.mean()),
                "peak_deviation": dev,
                "relief": relief,
                "psnr_coarse": p_coarse,
                "psnr_ours": p_ours,
                "psnr_wo_film": p_wof,
                "psnr_wo_cat": p_woc,
                "gain": float(p_ours - p_coarse) if np.isfinite(p_coarse) else np.nan,
                "abl_gap": float(p_ours - np.nanmax([p_wof, p_woc])),
                "ref_query_dist": rq,
                "lpips_ours": ours[key]["LPIPS"],
            })
        if (oi + 1) % 10 == 0:
            print(f"  scored {oi + 1}/{len(objs)} objects", flush=True)

    out = os.path.join(OUT_DIR, "descriptors.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\n{len(rows)} touches scored -> {out}")

    a = lambda k: np.array([r[k] for r in rows], float)
    for k in ("structure", "contact_frac", "relief", "psnr_ours", "gain", "abl_gap", "ref_query_dist"):
        v = a(k)
        print(f"  {k:16s} min {np.nanmin(v):8.3f}  median {np.nanmedian(v):8.3f}  max {np.nanmax(v):8.3f}")


if __name__ == "__main__":
    sys.exit(main())
