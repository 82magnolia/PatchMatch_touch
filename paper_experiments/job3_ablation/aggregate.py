"""Build the ablation table from the Job 3 runs.

Two families of ablation, both on the 20-object subset of the full-pipeline
benchmark:

  coarse-alignment arms   log/paper_job3_ablation/{arm}/{obj}/transfer/metrics.pkl
      modality at 4x: normal | RGB (color) | curvature | height
      scale at fixed modality: 1x (100) | 2x (50) | 4x (25)
      these change how the reference touch is aligned to the query

  network arms            log/paper_job3_refine_{arm}/metrics.json
      ours | w/o temporal FiLM | w/o normal concatenation
      "w/o network refinement" is the default arm's coarse number

Both coarse and refined numbers are reported for the alignment arms, so the
table shows whether an alignment choice still matters after the network runs.
"""
import json
import os
import pickle

import numpy as np

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
OUT_DIR = os.path.join(ROOT, "paper_experiments/job3_ablation")
KEYS = ["PSNR", "SSIM", "LPIPS", "MSE"]
DEFAULT_ARM = "mod_normal"          # surface normals at 4x == the default configuration

COARSE_ARMS = {
    "mod_normal": "Modality: surface normal (4x)  [default]",
    "mod_color": "Modality: RGB colour (4x)",
    "mod_curvature": "Modality: curvature (4x)",
    "mod_height": "Modality: height map (4x)",
    "scale_1x": "Scale: 1x sensor",
    "scale_2x": "Scale: 2x sensor",
    "scale_4x": "Scale: 4x sensor  [default, same run as mod_normal]",
}
REFINE_ARMS = {
    "ours": "Ours (full model)",
    "wo_film": "w/o temporal FiLM",
    "wo_cat": "w/o normal concatenation",
}


def subset_ids():
    with open(os.path.join(OUT_DIR, "subset_objects.txt")) as f:
        return [int(x) for x in f if x.strip()]


def coarse_arm(arm, ids):
    per_obj, missing = {}, []
    for obj in ids:
        p = os.path.join(ROOT, "log/paper_job3_ablation", arm, str(obj), "transfer", "metrics.pkl")
        if not os.path.exists(p):
            missing.append(obj)
            continue
        with open(p, "rb") as f:
            per_obj[obj] = {k: float(v) for k, v in pickle.load(f)["average"].items()}
    return per_obj, missing


def refine_arm(arm, coarse=DEFAULT_ARM):
    """Metrics for a network arm refining a given coarse alignment.

    The network arms must refine the SAME coarse transfer as the "w/o network
    refinement" row, or the comparison is meaningless. Newer runs are written to
    log/paper_job3_refine_{arm}_{coarse}; the un-suffixed path is the older
    layout and is only used as a fallback.
    """
    for rel in (f"log/paper_job3_refine_{arm}_{coarse}",
                f"log/paper_job3_refine_{arm}"):
        p = os.path.join(ROOT, rel, "metrics.json")
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            d["_source"] = rel
            return d
    return None


def summarise(per_obj):
    if not per_obj:
        return None
    return {k: float(np.mean([m[k] for m in per_obj.values()])) for k in KEYS}


def fmt(name, s, n, width=42):
    if s is None:
        return f"{name:{width}s} {'--':>4s}  (pending)"
    return (f"{name:{width}s} {n:4d} {s['PSNR']:7.2f} {s['SSIM']:7.4f} "
            f"{s['LPIPS']:7.4f} {s['MSE']:9.5f}")


def main():
    ids = subset_ids()
    out = {"subset_objects": ids, "coarse": {}, "refined": {}}

    print(f"Ablation subset: {len(ids)} objects "
          f"({ids[0]}..{ids[-1]}) of the full-pipeline benchmark\n")

    print("=== Coarse alignment (before network refinement) ===")
    print(f"{'Arm':42s} {'n':>4s} {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>7s} {'MSE':>9s}")
    print("-" * 82)
    for arm, label in COARSE_ARMS.items():
        per_obj, missing = coarse_arm(arm, ids)
        s = summarise(per_obj)
        out["coarse"][arm] = {"label": label, "n_objects": len(per_obj),
                              "missing": missing, "metrics": s}
        print(fmt(label, s, len(per_obj)))

    print("\n=== Network refinement (on the default coarse alignment) ===")
    print(f"{'Arm':42s} {'n':>4s} {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>7s} {'MSE':>9s}")
    print("-" * 82)
    # "w/o refinement" row = the default coarse arm.
    per_obj, _ = coarse_arm(DEFAULT_ARM, ids)
    s_none = summarise(per_obj)
    out["refined"]["wo_refinement"] = {"label": "w/o neural-network refinement",
                                       "n_objects": len(per_obj), "metrics": s_none}
    print(fmt("w/o neural-network refinement", s_none, len(per_obj)))
    for arm, label in REFINE_ARMS.items():
        d = refine_arm(arm)
        s = d["average"] if d else None
        n = d["n_objects"] if d else 0
        out["refined"][arm] = {"label": label, "n_objects": n, "metrics": s,
                               "source": d.get("_source") if d else None}
        print(fmt(label, s, n))

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    lines = ["% --- coarse-alignment ablations ---"]
    for arm, e in out["coarse"].items():
        m = e["metrics"]
        lines.append(f"{e['label']} & " + (
            f"{m['PSNR']:.2f} & {m['SSIM']:.4f} & {m['LPIPS']:.4f} \\\\" if m else "-- & -- & -- \\\\"))
    lines.append("% --- network ablations ---")
    for arm, e in out["refined"].items():
        m = e["metrics"]
        lines.append(f"{e['label']} & " + (
            f"{m['PSNR']:.2f} & {m['SSIM']:.4f} & {m['LPIPS']:.4f} \\\\" if m else "-- & -- & -- \\\\"))
    with open(os.path.join(OUT_DIR, "table_body.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved -> {OUT_DIR}/results.json and table_body.tex")


if __name__ == "__main__":
    main()
