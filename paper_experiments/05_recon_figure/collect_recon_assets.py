"""Gather everything behind the 3D-reconstruction figure into one folder.

Mirrors 04_paper_figures/collect_figure_assets.py, but for fig_recon: the four
rows of the reconstruction figure (reference touch, our prediction, the 3D
relief integrated from it, and the simulated colour image), plus the renders of
the object itself and the reference-shift sweep that chose which reference touch
to transfer from.

    python collect_recon_assets.py --tag 993_2_ref_cand11 --cand 11
"""
import argparse
import os
import pickle
import sys

import numpy as np

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
JOB = f"{ROOT}/log/paper_job05_recon_figure"
SHIFT = f"{ROOT}/log/paper_job04_paper_figures/shifted/993_ref2_shift"
QUERY_RENDER = f"{ROOT}/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini"
QUERY_PTS = f"{ROOT}/Taxim/results/object_folder_touch_query"
CKPT = f"{ROOT}/log/rebot_checkpoints_S_geomcat_film/best.pth"

ROW_NAMES = {"row1": "reference_touch", "row2": "our_prediction",
             "row3": "heightmap_3d", "row4": "simulated_rgb"}


def link(src, dst, man, root):
    if not os.path.exists(src):
        man.append(f"  MISSING  {os.path.relpath(dst, root)}  <-  {src}")
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)
    man.append(f"  {os.path.relpath(dst, root)}\n      -> {src}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", type=int, default=993)
    ap.add_argument("--query", type=int, default=2)
    ap.add_argument("--cand", type=int, default=11, help="the chosen shifted reference")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--alternates", nargs="*", default=["993_2_ref_cand13"])
    ap.add_argument("--out", default=f"{JOB}/figure_assets")
    args = ap.parse_args()

    tag = args.tag or f"{args.obj}_{args.query}_ref_cand{args.cand}"
    out = f"{args.out}/{tag}"
    os.makedirs(out, exist_ok=True)
    man = []

    # ---- the figure and its cells
    man.append("figure/")
    link(f"{JOB}/figure_{tag}.png", f"{out}/figure/stitched_preview.png", man, out)
    cells = sorted(os.listdir(f"{JOB}/assets/{tag}"))
    man.append("\ncells/  (one file per cell of the figure; columns are frames of the press)")
    for name in cells:
        row = next((k for k in ROW_NAMES if f"_{k}_" in name), None)
        col = name.split("_")[0]                    # col00, col01, ...
        frame = name.split("_")[1]                  # f007, f014, ...
        if row is None:
            continue
        link(f"{JOB}/assets/{tag}/{name}",
             f"{out}/cells/{ROW_NAMES[row]}/{col}_{frame}.png", man, out)
    for r in ROW_NAMES.values():
        n = len(os.listdir(f"{out}/cells/{r}")) if os.path.isdir(f"{out}/cells/{r}") else 0
        man.append(f"  cells/{r}/  {n} columns")

    # ---- the reference touch that was transferred, and how it was chosen
    man.append("\nreference_shift/  (the query is the benchmark's own touch; the reference "
               "was moved and re-simulated)")
    tdir = f"{JOB}/refsweep/cand{args.cand:02d}/{args.obj}/transfer"
    link(f"{tdir}/{args.query}_ref_tactile_normal.mp4",
         f"{out}/reference_shift/chosen_reference_touch_video.mp4", man, out)
    link(f"{tdir}/{args.query}_transferred.mp4",
         f"{out}/reference_shift/coarse_transfer_video.mp4", man, out)
    link(f"{tdir}/{args.query}_query_tactile_normal.mp4",
         f"{out}/reference_shift/ground_truth_query_video.mp4", man, out)
    for sc in ("100", "25"):
        link(f"{SHIFT}/{args.obj}/{args.cand}_scale{sc}_normal.jpg",
             f"{out}/reference_shift/chosen_reference_normal_scale{sc}.jpg", man, out)
        link(f"{QUERY_RENDER}/{args.obj}/{args.query}_scale{sc}_normal.jpg",
             f"{out}/reference_shift/query_normal_scale{sc}.jpg", man, out)
    link(f"{SHIFT}/contact_points.ply", f"{out}/reference_shift/shifted_contact_points.ply",
         man, out)
    link(f"{SHIFT}/labels.npy", f"{out}/reference_shift/shift_labels.npy", man, out)
    link(f"{QUERY_PTS}/{args.obj}/picked_points_query.ply",
         f"{out}/reference_shift/benchmark_query_points.ply", man, out)
    link(f"{JOB}/refsweep/candidates.png",
         f"{out}/reference_shift/sweep_contact_sheet.png", man, out)
    link(f"{JOB}/refsweep/candidates.pkl",
         f"{out}/reference_shift/sweep_scores.pkl", man, out)
    link(f"{tdir}/decomposition.pkl",
         f"{out}/reference_shift/alignment_decomposition.pkl", man, out)
    link(CKPT, f"{out}/reference_shift/refinement_checkpoint.pth", man, out)

    # ---- renders of the object itself
    man.append("\nobject_renders/")
    for sub in ("", "marked", "closeup"):
        src_dir = f"{JOB}/object_renders/{sub}".rstrip("/")
        if not os.path.isdir(src_dir):
            continue
        for name in sorted(os.listdir(src_dir)):
            if name.endswith(".png"):
                link(f"{src_dir}/{name}",
                     f"{out}/object_renders/{sub or 'views'}/{name}", man, out)

    # ---- other references that were tried
    man.append("\nalternates/")
    for alt in args.alternates:
        link(f"{JOB}/figure_{alt}.png", f"{out}/alternates/figure_{alt}.png", man, out)

    # ---- manifest
    recs = {r["cand"]: r for r in pickle.load(open(f"{JOB}/refsweep/candidates.pkl", "rb"))}
    c = recs[args.cand]
    ranked = sorted(recs.values(), key=lambda r: -r["psnr_refined"])
    head = [
        f"3D reconstruction figure - object {args.obj}, query touch {args.query}",
        "",
        f"  query               touch {args.query} of the ground-truth-retrieval benchmark, "
        f"unchanged",
        f"  reference           the benchmark's reference touch {2}, moved "
        f"{c['moved_mm']:.1f} mm across the surface (direction {c['direction_deg']} deg) "
        f"and re-simulated",
        f"  chosen from         {len(recs)} shifted references, ranked by how well the "
        f"prediction came out",
        f"  coarse transfer     {c['psnr_coarse']:.1f} dB",
        f"  refined             {c['psnr_refined']:.1f} dB   "
        f"(best of the sweep: {ranked[0]['psnr_refined']:.1f} dB, "
        f"worst: {ranked[-1]['psnr_refined']:.1f} dB)",
        f"  frames             {len(os.listdir(f'{out}/cells/reference_touch'))} columns, "
        f"spanning the in-contact part of the press",
        "",
        "Rows 3 and 4 are computed from row 2 alone: the predicted normals are integrated",
        "into a heightmap with a Poisson solver, and that heightmap is fed through Taxim's",
        "calibrated optical model. Neither uses ground truth.",
        "",
        "Everything below is a symlink.",
        "",
    ]
    with open(f"{out}/MANIFEST.txt", "w") as f:
        f.write("\n".join(head + man) + "\n")
    print("\n".join(head))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
