"""Gather everything one teaser figure is made of into a single folder of symlinks.

Handy when a figure is about to go into the paper: one directory holds the exact
images that were drawn, the videos and renders they came from, and the settings
that produced them, without copying a byte.

    python collect_figure_assets.py --tag pin7_951_11

Writes log/paper_job04_paper_figures/figure_assets/<tag>/ containing

    figure/      the rendered figure itself (PNG and PDF)
    drawn/       the twelve frames and two geometry renders actually placed in it,
                 named by where they sit in the figure. The two geometry renders
                 are real files, not links: the figure repaints their background
                 white and draws the red sensor box, and those steps happen at
                 drawing time, so the raw renders on disk do not look like what
                 the figure shows. The untouched originals are in sources/.
    frames/      every frame of the reference / predicted / coarse / ground-truth
                 sequences, in case a different set of six is wanted
    sources/     the videos, the original renders, the contact points, the
                 retrieval decision and the network checkpoint
    MANIFEST.txt what each link points to, and the numbers for this touch
"""
import argparse
import os
import pickle
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figlib import load_normal, sensor_box     # noqa: E402
from sweep_teaser import ASSETS, SOURCES        # noqa: E402

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
OUT = f"{ROOT}/log/paper_job04_paper_figures"
DEST = f"{OUT}/figure_assets"
CKPT = f"{ROOT}/log/rebot_checkpoints_S_geomcat_film/best.pth"


def link(src, dst, manifest):
    if not os.path.exists(src):
        manifest.append(f"  MISSING  {os.path.relpath(dst, DEST)}  <-  {src}")
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)
    manifest.append(f"  {os.path.relpath(dst, DEST)}\n      -> {src}")
    return True


def source_of(tag):
    """Which run a tag came from, longest prefix first so 'fpsh6_' beats 'fp'."""
    for key, cfg in sorted(SOURCES.items(), key=lambda kv: -len(kv[1]["prefix"])):
        if tag.startswith(cfg["prefix"]):
            return key, cfg, tag[len(cfg["prefix"]):]
    raise SystemExit(f"no run matches the tag {tag!r}; known prefixes: "
                     + ", ".join(c["prefix"] for c in SOURCES.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pin7_951_11")
    ap.add_argument("--frames", type=int, nargs="+", default=[4, 12, 21, 29, 38, 46],
                    help="the frames the figure draws (make_teaser.py's default)")
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()

    key, cfg, rest = source_of(args.tag)
    obj, pair = (int(x) for x in rest.split("_"))
    meta = pickle.load(open(f"{ASSETS}/{args.tag}_meta.pkl", "rb"))
    ref_idx = meta.get("pinned_ref", meta.get("ref_idx"))
    out = f"{DEST}/{args.tag}"
    os.makedirs(out, exist_ok=True)
    man = []

    # the figure itself
    man.append("figure/")
    for ext in ("png", "pdf"):
        link(f"{OUT}/teaser_{args.version}_{args.tag}.{ext}",
             f"{out}/figure/teaser_{args.version}_{args.tag}.{ext}", man)

    # exactly what is drawn, named by position in the figure
    man.append("\ndrawn/  (row 1 = given / reference, row 2 = predicted / query)")
    os.makedirs(f"{out}/drawn", exist_ok=True)
    for who, dst in (("refnorm", "row1_col0_reference_geometry_4x.png"),
                     ("querynorm", "row2_col0_query_geometry_4x.png")):
        src = f"{ASSETS}/{args.tag}_{who}_scale25.png"
        if not os.path.exists(src):
            man.append(f"  MISSING  drawn/{dst}  <-  {src}")
            continue
        path = f"{out}/drawn/{dst}"
        if os.path.islink(path):
            os.remove(path)
        cv2.imwrite(path, cv2.cvtColor(sensor_box(load_normal(src)), cv2.COLOR_RGB2BGR))
        man.append(f"  drawn/{dst}\n      (real file: {os.path.basename(src)} with the "
                   f"background repainted white and the 1x sensor box drawn)")
    for i, f in enumerate(args.frames, start=1):
        link(f"{ASSETS}/{args.tag}_ref_{f:03d}.png",
             f"{out}/drawn/row1_col{i}_reference_frame{f:03d}.png", man)
        link(f"{ASSETS}/{args.tag}_pred_{f:03d}.png",
             f"{out}/drawn/row2_col{i}_predicted_frame{f:03d}.png", man)

    # every frame, for re-picking which six to show
    man.append("\nframes/  (all frames of each sequence)")
    n = 0
    for kind, sub in (("ref", "reference"), ("pred", "predicted"),
                      ("coarse", "coarse_transfer"), ("gt", "ground_truth")):
        i = 0
        while os.path.exists(f"{ASSETS}/{args.tag}_{kind}_{i:03d}.png"):
            os.makedirs(f"{out}/frames/{sub}", exist_ok=True)
            dst = f"{out}/frames/{sub}/{i:03d}.png"
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            os.symlink(f"{ASSETS}/{args.tag}_{kind}_{i:03d}.png", dst)
            i += 1
            n += 1
        man.append(f"  frames/{sub}/  {i} frames")
    link(f"{ASSETS}/{args.tag}_meta.pkl", f"{out}/frames/meta.pkl", man)

    # where all of that came from
    man.append("\nsources/")
    tdir = f"{cfg['out']}/{obj}/transfer"
    link(f"{tdir}/{pair}_ref_tactile_normal.mp4",
         f"{out}/sources/reference_touch_video.mp4", man)
    link(f"{tdir}/{pair}_transferred.mp4", f"{out}/sources/coarse_transfer_video.mp4", man)
    link(f"{tdir}/{pair}_query_tactile_normal.mp4",
         f"{out}/sources/ground_truth_query_video.mp4", man)
    for sc in ("100", "25"):
        link(f"{cfg['ref_root']}/{obj}/{ref_idx}_scale{sc}_normal.jpg",
             f"{out}/sources/reference_normal_render_scale{sc}.jpg", man)
        link(f"{cfg['query_root']}/{obj}/{pair}_scale{sc}_normal.jpg",
             f"{out}/sources/query_normal_render_scale{sc}.jpg", man)
    link(f"{cfg['query_root']}/contact_points.ply", f"{out}/sources/contact_points.ply", man)
    link(f"{cfg['query_root']}/labels.npy", f"{out}/sources/shift_labels.npy", man)
    link(f"{ASSETS}/{args.tag}_refnorm_scale25.png",
         f"{out}/sources/reference_geometry_4x_as_rendered.png", man)
    link(f"{ASSETS}/{args.tag}_querynorm_scale25.png",
         f"{out}/sources/query_geometry_4x_as_rendered.png", man)
    for name in os.listdir(f"{cfg['out']}/{obj}"):
        if name.startswith("pinned_ref") and name.endswith(".tsv"):
            link(f"{cfg['out']}/{obj}/{name}", f"{out}/sources/{name}", man)
    link(f"{cfg['out']}/{obj}/retrieval/results.pkl",
         f"{out}/sources/retrieval_results.pkl", man)
    link(f"{tdir}/decomposition.pkl", f"{out}/sources/alignment_decomposition.pkl", man)
    link(CKPT, f"{out}/sources/refinement_checkpoint.pth", man)

    head = [
        f"Assets behind {OUT}/teaser_{args.version}_{args.tag}.png",
        "",
        f"  object              {obj}",
        f"  query touch         {pair}",
        f"  reference touch     {ref_idx}"
        + ("  (pinned, not retrieved)" if meta.get("pinned_ref") is not None else "  (retrieved)"),
        f"  frames drawn        {args.frames}  of {meta.get('n_frames')}",
        f"  coarse transfer     {meta.get('psnr_coarse', float('nan')):.1f} dB",
        f"  refined             {meta.get('psnr_refined', float('nan')):.1f} dB",
    ]
    if meta.get("moved_mm"):
        head.append(f"  query moved         {meta['moved_mm']:.1f} mm across the surface, "
                    f"then re-simulated")
    head += [
        f"  run                 {cfg['out']}",
        f"  benchmark           {meta.get('benchmark', '?')}",
        "",
        "Everything below is a symlink except the two geometry renders in drawn/,",
        "which are written out because the figure alters them before drawing.",
        "",
    ]
    with open(f"{out}/MANIFEST.txt", "w") as f:
        f.write("\n".join(head + man) + "\n")
    print("\n".join(head))
    print(f"{n} frame links + the drawn set and sources -> {out}")
    print(f"see {out}/MANIFEST.txt")


if __name__ == "__main__":
    main()
