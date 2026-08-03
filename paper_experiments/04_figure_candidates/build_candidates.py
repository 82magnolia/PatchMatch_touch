"""Pick and render candidate examples for the two figures this machine owns.

  fig_gt_retrieval  (paper_source/figures/fig_gt_retrieval.tex) -- 5 candidate
                    3 x 6 figures, i.e. 15 touch locations on 15 distinct objects
  fig_recon         (paper_source/figures/fig_recon.tex)        -- 5 candidate
                    4-row reconstruction strips, one per object

Selection uses the scores from score_touches.py rather than PSNR alone, because
the highest-PSNR touches are typically the ones where nothing visible happens.
Every candidate must clear hard admissibility gates first; the survivors are then
ranked by a weighted score and spread across distinct objects.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.join(HERE, "candidates.json")
OUTDIR = f"{ROOT}/log/paper_fig_candidates"

JOB2 = f"{ROOT}/log/paper_job02_gt_retrieval_figure_normalmatch"
MAKE_FIG = f"{ROOT}/paper_experiments/02_gt_retrieval_figure/make_figure.py"
MAKE_RECON = f"{ROOT}/paper_experiments/03_recon_visuotactile/make_recon_figure.py"

PY = sys.executable


def norm(v):
    """Min-max to [0,1]; constant inputs map to 0.5."""
    v = np.asarray(v, dtype=float)
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-9:
        return np.full_like(v, 0.5)
    return (v - lo) / (hi - lo)


# --- Admissibility gates -----------------------------------------------------
# These are absolute requirements, not preferences: a touch failing any of them
# makes a bad figure panel no matter how good its other numbers are.
GATES = {
    "gt_retrieval": dict(
        contact=(0.45, None),      # the contact must actually be visible
        refined_psnr=(26.0, None),  # our prediction must be good, or the figure
                                    # argues against us
        gain=(3.0, None),           # refinement must visibly do something, else
                                    # columns 4 and 5 look identical
        cover_ref=(0.08, 0.85),     # the 4x render must show surface, but not be
        cover_query=(0.08, 0.85),   # one edge-to-edge flat wall
        pose_diff=(0.04, None),     # query pose must differ from reference, or
                                    # the analogy looks trivial
    ),
    "recon": dict(
        contact=(0.50, None),
        refined_psnr=(27.0, None),
        temporal=(0.65, None),      # contact must grow/shrink across the press,
                                    # otherwise every column looks the same
        structure=(0.006, None),    # needs shape, or the 3D relief is a blob
        cover_query=(0.08, 0.90),
    ),
}

# --- Ranking weights ---------------------------------------------------------
WEIGHTS = {
    "gt_retrieval": dict(structure=0.30, gain=0.25, contact=0.20,
                         pose_diff=0.15, refined_psnr=0.10),
    "recon": dict(temporal=0.30, structure=0.30, contact=0.20,
                  refined_psnr=0.20),
}


def passes(rec, gates):
    for key, (lo, hi) in gates.items():
        v = rec[key]
        if lo is not None and v < lo:
            return False
        if hi is not None and v > hi:
            return False
    return True


def rank(cands, kind):
    pool = [c for c in cands if passes(c, GATES[kind])]
    if not pool:
        return []
    cols = {k: norm([c[k] for c in pool]) for k in WEIGHTS[kind]}
    for i, c in enumerate(pool):
        c["score"] = float(sum(w * cols[k][i] for k, w in WEIGHTS[kind].items()))
    return sorted(pool, key=lambda c: -c["score"])


def spread_by_object(ranked, n):
    """Take the n best entries, at most one per object."""
    out, seen = [], set()
    for c in ranked:
        if c["obj"] in seen:
            continue
        out.append(c)
        seen.add(c["obj"])
        if len(out) == n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_figs", type=int, default=5)
    ap.add_argument("--rows", type=int, default=3, help="rows per gt_retrieval figure")
    ap.add_argument("--skip_recon", action="store_true")
    ap.add_argument("--skip_gt", action="store_true")
    args = ap.parse_args()

    cands = json.load(open(CAND))
    os.makedirs(f"{OUTDIR}/gt_retrieval", exist_ok=True)
    os.makedirs(f"{OUTDIR}/recon", exist_ok=True)
    # Merge into any existing manifest so that running with --skip_gt / --skip_recon
    # refreshes only one section instead of wiping the other.
    manifest = {"gt_retrieval": [], "recon": []}
    mpath = f"{OUTDIR}/manifest.json"
    if os.path.exists(mpath):
        try:
            manifest.update(json.load(open(mpath)))
        except (ValueError, OSError):
            pass
    if not args.skip_gt:
        manifest["gt_retrieval"] = []
    if not args.skip_recon:
        manifest["recon"] = []

    # ---------------- fig_gt_retrieval ----------------
    if not args.skip_gt:
        ranked = rank(cands, "gt_retrieval")
        print(f"[gt_retrieval] {len(ranked)}/{len(cands)} touches pass the gates")
        need = args.n_figs * args.rows
        picks = spread_by_object(ranked, need)
        print(f"[gt_retrieval] selected {len(picks)} touches on distinct objects")
        for f in range(args.n_figs):
            group = picks[f * args.rows:(f + 1) * args.rows]
            if len(group) < args.rows:
                break
            keys = [f"{c['obj']}_{c['pair']}" for c in group]
            out = f"{OUTDIR}/gt_retrieval/candidate_{f+1}.png"
            cmd = [PY, MAKE_FIG, "--base", JOB2, "--touches", *keys,
                   "--n_rows", str(args.rows), "--out", out]
            print(f"  candidate {f+1}: {keys}")
            subprocess.run(cmd, check=True, cwd=ROOT,
                           stdout=subprocess.DEVNULL)
            manifest["gt_retrieval"].append(
                dict(candidate=f + 1, out=out, touches=group))

    # ---------------- fig_recon ----------------
    if not args.skip_recon:
        ranked = rank(cands, "recon")
        print(f"[recon] {len(ranked)}/{len(cands)} touches pass the gates")
        picks = spread_by_object(ranked, args.n_figs)
        for i, c in enumerate(picks, 1):
            print(f"  candidate {i}: {c['obj']}_{c['pair']}  score {c['score']:.3f}")
            subprocess.run(
                [PY, MAKE_RECON, "--obj", str(c["obj"]), "--pair", str(c["pair"]),
                 "--n_cols", "6", "--out_dir", f"{OUTDIR}/recon"],
                check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
            manifest["recon"].append(
                dict(candidate=i, obj=c["obj"], pair=c["pair"],
                     out=f"{OUTDIR}/recon/figure_{c['obj']}_{c['pair']}.png", rec=c))

    with open(f"{OUTDIR}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print("\nwrote", f"{OUTDIR}/manifest.json")


if __name__ == "__main__":
    main()
