"""Candidates for fig_full_pipeline: methods (rows) x video frames (columns).

Scored on the full-pipeline benchmark, where retrieval is part of the system, so
this is the figure that shows the complete method against the baselines.

A good candidate needs BOTH:
  * visible structure in the ground truth, so the reader can judge the shapes, and
  * a real margin for our method over the best baseline on that touch, so the
    figure is representative of the quantitative table rather than a lucky pick.

Rows follow paper_source/figures/fig_full_pipeline.tex. The quilting baseline is
shown as a real press sequence (paper_experiments/baselines/quilting_video.py
re-renders its quilted relief through the simulator's press profile) rather than
one tiled image, so it is no longer marked "N/A (image only)". TaRF is omitted --
it was withdrawn from the results.
"""
import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
import torch

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
sys.path.insert(0, os.path.join(ROOT, "paper_experiments/figures"))
from make_candidates import (cell, save, read_video, z, W, H, FLAT,   # noqa: E402
                             get_model, CKPTS)
sys.path.insert(0, os.path.join(ROOT, "rebot_net"))
from dataset import TactileTransferDataset                            # noqa: E402

J2_PIPE = os.path.join(ROOT, "log/paper_job2_pipeline_normal")
J2_COND = os.path.join(ROOT, "Taxim/results/gen_contact_raw_eval_tactile_normal_pseudo_mini")
BASE = os.path.join(ROOT, "log/paper_job2_baselines")
OUT = os.path.join(ROOT, "log/paper_figure_candidates/full_pipeline")
N_COLS = 8


class NestedDS(TactileTransferDataset):
    def __init__(self, *a, **k):
        self.NUM_PAIRS = 32
        super().__init__(*a, **k)

    def _obj_dir(self, obj_id):
        return os.path.join(self.transfer_dir, str(obj_id), "transfer")


def inr_content(obj, t):
    """How much the ObjectFolder INR baseline actually draws for this touch.

    Returned as the largest within-frame variation over the clip, averaged over the
    three colour channels. A frame that is one flat colour scores 0 no matter which
    colour it is, which is what we want: a blank INR row makes the comparison
    figure pointless. In this benchmark the flattest touches score ~0.0003 and the
    most detailed ~0.024.
    """
    v = read_video(os.path.join(BASE, "inr", str(obj), "transfer", f"{t}_transferred.mp4"))
    if not v:
        return 0.0
    return float(max(f.reshape(-1, 3).std(0).mean() for f in v))


def per_touch(path, key):
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f).get(key, {})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5,
                    help="how many candidates to keep, in score order")
    ap.add_argument("--redo", action="store_true",
                    help="re-render candidates whose panels are already on disk")
    ap.add_argument("--n-cols", type=int, default=N_COLS,
                    help="frames shown per method, sampled across the in-contact span")
    ap.add_argument("--min-inr-content", type=float, default=0.010,
                    help="reject a touch whose ObjectFolder INR video is blank; this is the "
                         "smallest within-frame variation the INR row may have (0 is a "
                         "single flat colour, the best touches reach about 0.024)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(ROOT, "log/paper_job2_refine_ours_normal/metrics.json")) as f:
        ours = json.load(f)["per_touch"]

    rows = []
    for key, m in ours.items():
        obj, t = key.split("_")
        obj, t = int(obj), int(t)
        coarse = per_touch(os.path.join(J2_PIPE, str(obj), "transfer", "metrics.pkl"), "per_touch")
        qui = per_touch(os.path.join(BASE, "quilting", str(obj), "transfer", "metrics.pkl"), "per_touch")
        inr = per_touch(os.path.join(BASE, "inr", str(obj), "transfer", "metrics.pkl"), "per_touch")
        b = [v.get("PSNR", np.nan) for v in (qui.get(t, {}), inr.get(t, {}))]
        if not np.isfinite(b).any():
            continue
        gt = read_video(os.path.join(J2_PIPE, str(obj), "transfer", f"{t}_query_tactile_normal.mp4"))
        if gt is None:
            continue
        dev = np.array([np.linalg.norm(2 * g - 1 - FLAT, axis=-1).mean() for g in gt])
        k = int(np.argmax(dev))
        rows.append({"object": obj, "touch": t, "peak": k, "n": len(gt),
                     "structure": float(gt[k].std()),
                     "psnr_ours": m["PSNR"],
                     "best_baseline": float(np.nanmax(b)),
                     "margin": float(m["PSNR"] - np.nanmax(b)),
                     "inr_content": inr_content(obj, t),
                     "psnr_coarse": float(coarse.get(t, {}).get("PSNR", np.nan))})

    score = z([r["structure"] for r in rows]) + z([r["margin"] for r in rows])
    by_key = {(r["object"], r["touch"]): (r, float(score[i])) for i, r in enumerate(rows)}

    # Candidates already rendered stay in the figure; new picks are appended below
    # them, so asking for more options never reshuffles the ones already reviewed.
    prev_path = os.path.join(ROOT, "log/paper_figure_candidates/candidates.json")
    prev = (json.load(open(prev_path)).get("full_pipeline", [])
            if os.path.exists(prev_path) else [])
    picked, seen = [], set()
    for c in prev[:args.n]:
        key = (c["object"], c["touch"])
        if key in by_key:
            picked.append(by_key[key])
            seen.add(c["object"])

    n_kept = len(picked)
    skipped_blank = 0
    for i in np.argsort(-score):
        if len(picked) >= args.n:
            break
        r = rows[i]
        if r["object"] in seen:
            continue
        # the point of this figure is the comparison, so a baseline row that is one
        # flat colour makes the candidate useless however good our own numbers are
        if r["inr_content"] < args.min_inr_content:
            skipped_blank += 1
            continue
        seen.add(r["object"])
        picked.append((r, float(score[i])))
    print(f"kept {n_kept} existing, added {len(picked) - n_kept} new "
          f"({skipped_blank} passed over for a blank INR row)")

    cands = []
    for r, s in picked:
        o, t = r["object"], r["touch"]
        d = os.path.join(OUT, f"{o}_{t}")
        if not args.redo and os.path.exists(os.path.join(d, "preview.png")):
            cands.append({"object": o, "touch": t, "score": s, "structure": r["structure"],
                          "psnr_ours": r["psnr_ours"], "best_baseline": r["best_baseline"],
                          "margin": r["margin"], "inr_content": r["inr_content"],
                          "preview": f"log/paper_figure_candidates/full_pipeline/{o}_{t}/preview.png"})
            print(f"  full_pipeline {o}_{t}  already rendered, kept")
            continue
        ds = NestedDS(J2_PIPE, [o], split="test", cond_dir=J2_COND,
                      film_modality="normal", film_scale=100, geom_concat=True,
                      video_type="tactile_normal", time_cond="film")
        model, _ = get_model("ours", device)
        preds, gts, lqs = [], [], []
        with torch.no_grad():
            for lq, gt, _b, film, tn in ds.iter_video_pairs(o, t):
                pr = model(lq.unsqueeze(0).to(device), film=None,
                           t=torch.tensor([tn], device=device)).squeeze(0)
                preds.append(pr.cpu().clamp(0, 1).permute(1, 2, 0).numpy())
                gts.append(gt.permute(1, 2, 0).numpy())
                lqs.append(lq[1, :3].permute(1, 2, 0).numpy())
        if not preds:
            continue

        tdir = os.path.join(J2_PIPE, str(o), "transfer")
        ref_v = read_video(os.path.join(tdir, f"{t}_ref_tactile_normal.mp4"))
        qui_seq = os.path.join(ROOT, "log/paper_job2_baselines/quilting_video",
                               str(o), "transfer", f"{t}_transferred.mp4")
        qui_v = read_video(qui_seq) or read_video(
            os.path.join(BASE, "quilting", str(o), "transfer", f"{t}_transferred.mp4"))
        qui_is_still = not os.path.exists(qui_seq)
        inr_v = read_video(os.path.join(BASE, "inr", str(o), "transfer", f"{t}_transferred.mp4"))
        # Sample columns from the part of the press that is actually in contact.
        # Spreading them over the whole clip put the flat pre- and post-contact
        # frames in the first and last column, which made every row -- not just the
        # INR one -- open and close on a blank panel.
        dev = np.array([np.linalg.norm(2 * g - 1 - FLAT, axis=-1).mean() for g in gts])
        idx = np.where(dev > dev.max() * 0.45)[0]
        if len(idx) < args.n_cols:
            idx = np.arange(len(gts))
        cols = np.linspace(idx[0], idx[-1], args.n_cols).round().astype(int)

        spec = [("reference", ref_v, False), ("quilting", qui_v, qui_is_still),
                ("inr", inr_v, False), ("ours_coarse", lqs, False),
                ("ours_refined", preds, False), ("gt", gts, False)]
        label = {"reference": "Reference tactile normal", "quilting": "Quilting (press sequence)",
                 "inr": "ObjectFolder INR", "ours_coarse": "Ours: coarse",
                 "ours_refined": "Ours: refined", "gt": "Ground truth"}
        d = os.path.join(OUT, f"{o}_{t}")
        grid = []
        for nm, seq, image_only in spec:
            line = []
            for ci, fi in enumerate(cols):
                if seq is None:
                    im, cap = None, "missing"
                elif image_only and ci > 0:
                    im, cap = None, "N/A (image only)"
                else:
                    im = seq[min(fi, len(seq) - 1)]
                    cap = f"{label[nm]}  f{fi}"
                c = cell(im, cap)
                line.append(c)
                if im is not None:
                    save(c, os.path.join(d, f"{nm}_c{ci}.png"))
            grid.append(np.hstack(line))
        save(np.vstack(grid), os.path.join(d, "preview.png"))
        cands.append({"object": o, "touch": t, "score": s, "structure": r["structure"],
                      "psnr_ours": r["psnr_ours"], "best_baseline": r["best_baseline"],
                      "margin": r["margin"], "inr_content": r["inr_content"],
                      "preview": f"log/paper_figure_candidates/full_pipeline/{o}_{t}/preview.png"})
        print(f"  full_pipeline {o}_{t}  margin {r['margin']:.1f} dB over best baseline"
              f"  inr_content {r['inr_content']:.4f}")

    p = os.path.join(ROOT, "log/paper_figure_candidates/candidates.json")
    ex = json.load(open(p)) if os.path.exists(p) else {}
    ex["full_pipeline"] = cands
    with open(p, "w") as f:
        json.dump(ex, f, indent=2)
    print(f"\nSaved -> {p}")


if __name__ == "__main__":
    sys.exit(main())
