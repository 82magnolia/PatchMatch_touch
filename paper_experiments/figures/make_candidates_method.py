"""Candidates for fig_method panel (b): the coarse-alignment illustration.

Panel (b) needs a touch where SuperPoint/SuperGlue actually found clean
correspondences between the reference and query normal maps, so the reader can
see real matches and a sensible homography rather than a degenerate fit.

Every transferred object already cached its correspondence diagnostics in
{obj}/decomposition.pkl (linear_matches, linear_inliers, valid_fraction, and
whether the offset stage had to be zeroed). We rank on those, then re-run the
transfer for the winners with --save_match_figures to render the actual
ref|query match panel, which the runs did not save by default.
"""
import json
import os
import pickle
import subprocess
import sys

import numpy as np

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
PY = "/home/junhokim/miniconda3/envs/pm_touch/bin/python"
TRANSFER = os.path.join(ROOT, "log/paper_job1_transfer_normal")
REF_BASE = os.path.join(ROOT, "Taxim/results/gen_contact_full_tactile_normal_pseudo_mini")
QUERY_BASE = os.path.join(ROOT, "Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini")
RETRIEVAL = os.path.join(ROOT, "log/touch_retrieval")
OUT = os.path.join(ROOT, "log/paper_figure_candidates/method")


def main():
    rows = []
    for d in sorted(os.listdir(TRANSFER)):
        p = os.path.join(TRANSFER, d, "decomposition.pkl")
        if not d.isdigit() or not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            dec = pickle.load(f)
        cm = os.path.join(TRANSFER, d, "metrics.pkl")
        coarse = pickle.load(open(cm, "rb"))["per_touch"] if os.path.exists(cm) else {}
        for t, e in dec.items():
            if e.get("transform_type") != "homography":
                continue
            zeroed = "zero" in str(e.get("offset_status", "")).lower()
            rows.append({
                "object": int(d), "touch": int(t),
                "matches": e.get("linear_matches", 0),
                "inliers": e.get("linear_inliers", 0),
                "valid_fraction": e.get("valid_fraction", 0.0),
                "offset_zeroed": bool(zeroed),
                "psnr_coarse": float(coarse.get(int(t), {}).get("PSNR", np.nan)),
            })

    # Want many inliers, a high inlier ratio, most of the warp in bounds, a
    # successful offset stage, and a coarse result that actually looks right.
    def sc(r):
        ratio = r["inliers"] / max(r["matches"], 1)
        return (0.05 * r["inliers"] + 2.0 * ratio + 2.0 * r["valid_fraction"]
                + (0.0 if r["offset_zeroed"] else 1.5)
                + 0.05 * (r["psnr_coarse"] if np.isfinite(r["psnr_coarse"]) else 0))

    rows.sort(key=sc, reverse=True)
    picked, seen = [], set()
    for r in rows:
        if r["object"] in seen:
            continue
        seen.add(r["object"])
        picked.append(r)
        if len(picked) == 5:
            break

    os.makedirs(OUT, exist_ok=True)
    cands = []
    for r in picked:
        o, t = r["object"], r["touch"]
        save_dir = os.path.join(OUT, f"{o}_{t}")
        os.makedirs(save_dir, exist_ok=True)
        cmd = [PY, os.path.join(ROOT, "main_retrieval_transfer_feat_match.py"),
               "--query_dir", os.path.join(QUERY_BASE, str(o)),
               "--ref_dir", os.path.join(REF_BASE, str(o)),
               "--retrieval_pkl", os.path.join(RETRIEVAL, str(o), "results.pkl"),
               "--modality", "normal", "--video_type", "tactile_normal",
               "--video_scale", "100.", "--match_scale", "25.",
               "--match_scale_convention", "obj_scale_factor",
               "--matcher", "superpoint_superglue",
               "--offset_matcher", "superpoint_superglue",
               "--offset_method", "median",
               "--save_dir", save_dir, "--no_nnf_figures",
               "--save_match_figures"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="6")
        res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
        figs = [f for f in os.listdir(save_dir) if "match" in f.lower() and f.endswith((".png", ".jpg"))]
        cands.append({
            "object": o, "touch": t, "matches": r["matches"], "inliers": r["inliers"],
            "valid_fraction": r["valid_fraction"], "offset_zeroed": r["offset_zeroed"],
            "psnr_coarse": r["psnr_coarse"],
            "match_figures": sorted(figs)[:3],
            "dir": f"log/paper_figure_candidates/method/{o}_{t}",
            "ok": res.returncode == 0,
        })
        print(f"  method {o}_{t}: {r['inliers']}/{r['matches']} inliers, "
              f"valid {r['valid_fraction']:.2f}, {len(figs)} match figure(s), rc={res.returncode}")

    p = os.path.join(ROOT, "log/paper_figure_candidates/candidates.json")
    ex = json.load(open(p)) if os.path.exists(p) else {}
    ex["method"] = cands
    with open(p, "w") as f:
        json.dump(ex, f, indent=2)
    print(f"\nSaved -> {p}")


if __name__ == "__main__":
    sys.exit(main())
