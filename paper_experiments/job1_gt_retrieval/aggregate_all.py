"""Collect every Job 1 method into one comparison table.

Sources
  ours (coarse)   log/transfer_feat_match_.../{obj}/metrics.pkl      per-object 'average'
  ours (refined)  log/paper_job1_refine_ours/metrics.pkl             per-object dict
  w/o temporal FiLM / w/o normal concat   same shape, sibling dirs
  quilting / inr  log/paper_job1_baselines/{m}/{obj}/transfer/metrics.pkl

Every source ultimately averages MSE / PSNR / SSIM / LPIPS per touch, then per
object, then over objects -- so all rows are comparable. Writes results.json and
a LaTeX table body ready to paste into the paper.
"""
import json
import os
import pickle

import numpy as np

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
OUT_DIR = os.path.join(ROOT, "paper_experiments/job1_gt_retrieval")
EVAL_IDS = list(range(951, 1001))
KEYS = ["PSNR", "SSIM", "LPIPS", "MSE"]

# The paper's coarse alignment is surface normals at 4x the sensor footprint.
# The "_normal" runs use that (rebuilt by run_transfer_normal.sh); the curvature
# rows are the original transfer the refinement checkpoints were TRAINED on, kept
# so the table shows what that train/test mismatch costs.
PER_OBJECT_PKL = {
    "Ours (refined, normals)": "log/paper_job1_refine_ours_normal/metrics.pkl",
    "w/o temporal FiLM": "log/paper_job1_refine_wo_film_normal/metrics.pkl",
    "w/o normal concatenation": "log/paper_job1_refine_wo_normalcat_normal/metrics.pkl",
    "Ours (refined, curvature)": "log/paper_job1_refine_ours/metrics.pkl",
}
PER_OBJECT_DIR = {
    "Ours (coarse transfer, normals)": "log/paper_job1_transfer_normal/{obj}/metrics.pkl",
    "Ours (coarse transfer, curvature)": "log/transfer_feat_match_pseudo_mini_tactile_normal_superpoint_superglue/{obj}/metrics.pkl",
    "Tactile Normal Quilting": "log/paper_job1_baselines/quilting/{obj}/transfer/metrics.pkl",
    "ObjectFolder INR": "log/paper_job1_baselines/inr/{obj}/transfer/metrics.pkl",
}


def from_pkl(rel):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        return None, "not run yet"
    with open(path, "rb") as f:
        d = pickle.load(f)
    per_obj = {int(k): {kk: float(vv) for kk, vv in v.items()}
               for k, v in d["per_object"].items()}
    return per_obj, None


def from_dirs(tmpl):
    per_obj, missing = {}, []
    for obj in EVAL_IDS:
        path = os.path.join(ROOT, tmpl.format(obj=obj))
        if not os.path.exists(path):
            missing.append(obj)
            continue
        with open(path, "rb") as f:
            d = pickle.load(f)
        per_obj[obj] = {k: float(v) for k, v in d["average"].items()}
    note = f"{len(missing)} of {len(EVAL_IDS)} objects missing" if missing else None
    return per_obj, note


def summarise(per_obj):
    return {k: float(np.mean([m[k] for m in per_obj.values()])) for k in KEYS}


def main():
    rows = {}
    for name, rel in PER_OBJECT_PKL.items():
        per_obj, note = from_pkl(rel)
        rows[name] = {"per_object": per_obj, "note": note}
    for name, tmpl in PER_OBJECT_DIR.items():
        per_obj, note = from_dirs(tmpl)
        rows[name] = {"per_object": per_obj, "note": note}

    table = {}
    for name, r in rows.items():
        if not r["per_object"]:
            table[name] = {"status": r["note"] or "no data", "n_objects": 0}
            continue
        table[name] = {"n_objects": len(r["per_object"]),
                       "note": r["note"],
                       **summarise(r["per_object"])}

    order = ["Tactile Normal Quilting", "ObjectFolder INR",
             "Ours (coarse transfer, normals)", "Ours (refined, normals)",
             "Ours (coarse transfer, curvature)", "Ours (refined, curvature)",
             "w/o temporal FiLM", "w/o normal concatenation"]

    print(f"{'Method':30s} {'n':>4s} {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>7s} {'MSE':>9s}")
    print("-" * 70)
    for name in order:
        t = table.get(name, {})
        if not t.get("n_objects"):
            print(f"{name:30s} {'-':>4s}   {t.get('status', 'pending')}")
            continue
        print(f"{name:30s} {t['n_objects']:4d} {t['PSNR']:7.2f} {t['SSIM']:7.4f} "
              f"{t['LPIPS']:7.4f} {t['MSE']:9.5f}")

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump({"benchmark": "ground-truth retrieval",
                   "eval_objects": f"{EVAL_IDS[0]}-{EVAL_IDS[-1]}",
                   "table": table,
                   "per_object": {k: v["per_object"] for k, v in rows.items()}}, f, indent=2)

    # LaTeX body (empty cells are left as -- so the row is still visible)
    lines = []
    for name in order:
        t = table.get(name, {})
        if not t.get("n_objects"):
            lines.append(f"{name} & -- & -- & -- \\\\")
        else:
            lines.append(f"{name} & {t['PSNR']:.2f} & {t['SSIM']:.4f} & {t['LPIPS']:.4f} \\\\")
    with open(os.path.join(OUT_DIR, "table_body.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved -> {OUT_DIR}/results.json and table_body.tex")


if __name__ == "__main__":
    main()
