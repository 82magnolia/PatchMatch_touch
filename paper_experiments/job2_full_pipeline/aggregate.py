"""Collect the full-pipeline benchmark comparison.

Unlike Job 1, retrieval is part of the system under test here: for each object,
3-4 held-out query touches are matched against the object's remaining touches by
DINOv3, and the retrieved reference is what gets transferred. All methods see the
same split and the same retrieval, so the table isolates the prediction stage.

  ours (coarse)   log/paper_job2_pipeline/{obj}/transfer/metrics.pkl
  ours (refined)  log/paper_job2_refine_ours/metrics.json
  quilting / inr  log/paper_job2_baselines/{m}/{obj}/transfer/metrics.pkl
"""
import json
import os
import pickle

import numpy as np

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
OUT_DIR = os.path.join(ROOT, "paper_experiments/job2_full_pipeline")
KEYS = ["PSNR", "SSIM", "LPIPS", "MSE"]

# The method's default coarse alignment is surface normals at 4x; the curvature
# rows are kept alongside because the refinement checkpoints were TRAINED on a
# curvature-modality transfer, so the pair measures that train/test mismatch.
DIR_SOURCES = {
    "Tactile Normal Quilting": "log/paper_job2_baselines/quilting/{obj}/transfer/metrics.pkl",
    "ObjectFolder INR": "log/paper_job2_baselines/inr/{obj}/transfer/metrics.pkl",
    # Three trained TaRF diffusion checkpoints, all run with the float32
    # conditioning-encoder fix so the rows are comparable with each other.
    "TaRF (epoch 5, finetuned)": "log/paper_job2_baselines/tarf/{obj}/transfer/metrics.pkl",
    "TaRF (epoch 29, from scratch)": "log/paper_job2_baselines/tarf_v2/{obj}/transfer/metrics.pkl",
    "TaRF (epoch 29, finetuned)": "log/paper_job2_baselines/tarf_v3/{obj}/transfer/metrics.pkl",
    "Ours (coarse transfer, normals)": "log/paper_job2_pipeline_normal/{obj}/transfer/metrics.pkl",
    "Ours (coarse transfer, curvature)": "log/paper_job2_pipeline/{obj}/transfer/metrics.pkl",
}
JSON_SOURCES = {
    "Ours (refined, normals)": "log/paper_job2_refine_ours_normal/metrics.json",
    "Ours (refined, curvature)": "log/paper_job2_refine_ours/metrics.json",
}
ORDER = ["Tactile Normal Quilting", "ObjectFolder INR",
         "TaRF (epoch 5, finetuned)", "TaRF (epoch 29, from scratch)",
         "TaRF (epoch 29, finetuned)",
         "Ours (coarse transfer, normals)", "Ours (refined, normals)",
         "Ours (coarse transfer, curvature)", "Ours (refined, curvature)"]


def object_ids():
    with open(os.path.join(OUT_DIR, "splits.json")) as f:
        return sorted(int(k) for k in json.load(f)["objects"])


def from_dirs(tmpl, ids):
    per_obj, missing = {}, []
    for obj in ids:
        p = os.path.join(ROOT, tmpl.format(obj=obj))
        if not os.path.exists(p):
            missing.append(obj)
            continue
        with open(p, "rb") as f:
            per_obj[obj] = {k: float(v) for k, v in pickle.load(f)["average"].items()}
    return per_obj, missing


def from_json(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return {}, None
    with open(p) as f:
        d = json.load(f)
    return {int(k): v for k, v in d["per_object"].items()}, d


def summarise(per_obj):
    if not per_obj:
        return None
    return {k: float(np.mean([m[k] for m in per_obj.values()])) for k in KEYS}


def main():
    ids = object_ids()
    table, per_object_out = {}, {}

    for name, tmpl in DIR_SOURCES.items():
        per_obj, missing = from_dirs(tmpl, ids)
        per_object_out[name] = per_obj
        table[name] = {"n_objects": len(per_obj), "n_missing": len(missing),
                       "metrics": summarise(per_obj)}
    for name, rel in JSON_SOURCES.items():
        per_obj, _ = from_json(rel)
        per_object_out[name] = per_obj
        table[name] = {"n_objects": len(per_obj), "n_missing": len(ids) - len(per_obj),
                       "metrics": summarise(per_obj)}

    print(f"Full-pipeline benchmark: {len(ids)} objects\n")
    print(f"{'Method':34s} {'n':>4s} {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>7s} {'MSE':>9s}")
    print("-" * 74)
    for name in ORDER:
        t = table.get(name, {})
        m = t.get("metrics")
        if not m:
            print(f"{name:34s} {'-':>4s}   pending ({t.get('n_missing', '?')} missing)")
            continue
        print(f"{name:34s} {t['n_objects']:4d} {m['PSNR']:7.2f} {m['SSIM']:7.4f} "
              f"{m['LPIPS']:7.4f} {m['MSE']:9.5f}")

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump({"benchmark": "full pipeline", "n_objects": len(ids),
                   "table": table, "per_object": per_object_out}, f, indent=2)

    lines = []
    for name in ORDER:
        m = table.get(name, {}).get("metrics")
        lines.append(f"{name} & " + (
            f"{m['PSNR']:.2f} & {m['SSIM']:.4f} & {m['LPIPS']:.4f} \\\\" if m else "-- & -- & -- \\\\"))
    with open(os.path.join(OUT_DIR, "table_body.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved -> {OUT_DIR}/results.json and table_body.tex")


if __name__ == "__main__":
    main()
