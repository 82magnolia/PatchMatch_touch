"""Write out the individual pictures a method figure is built from.

make_method.py draws everything into one page. This script saves the same
pieces as separate image files, exactly as they appear in the figure -- normal
renders already repainted onto white -- so they can be dropped into slides or a
hand-made layout without re-deriving anything.

Files are named by the role they play in the figure, not by their position in
the asset cache, and a manifest.json plus a README.txt record what each one is
and which numbers go with it. The feature correspondences are written as CSV
rather than as an image, since the figure draws them as lines rather than
pasting a picture.

  python export_method_assets.py --tag 984_6 --ret_tag bench984_6 --mid 25
"""
import argparse
import csv
import json
import os
import shutil

import cv2
import numpy as np

from figlib import ASSETS, load, load_normal, sensor_box
from make_method import OUT, figure_inputs


def save(path, rgb):
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def export(d, out_dir, keep_raw):
    ret, md, tag, mid, gap = d["ret"], d["md"], d["tag"], d["mid"], d["gap"]
    os.makedirs(out_dir, exist_ok=True)
    man = {"touch": {"object": int(md["obj"]), "index": int(md["pair"])},
           "frame_index_used_for_stills": mid, "files": {}}

    def note(name, what, **extra):
        man["files"][name] = dict(what=what, **extra)

    # --- step 1: the query location and the reference thumbnails ------------
    save(f"{out_dir}/step1_query_location.png", load_normal(ret["qimg"]))
    note("step1_query_location.png",
         "the query location's geometry render at 1x the sensor, as drawn in step 1")

    refs = []
    for rank, (idx, sim) in enumerate(ret["show"]):
        top1 = idx == ret["top1"]
        stem = f"step1_ref{rank}_touch{idx}" + ("_bestmatch" if top1 else "")
        save(f"{out_dir}/{stem}_geometry.png",
             load_normal(f"{ASSETS}/{ret['tag']}_db{idx}_normal.png"))
        save(f"{out_dir}/{stem}_touch.png",
             load(f"{ASSETS}/{ret['tag']}_db{idx}_touch.png"))
        note(f"{stem}_geometry.png",
             "reference touch geometry at 1x the sensor (top row of step 1)",
             touch_index=int(idx), similarity_to_query=round(sim, 4),
             is_best_match=top1)
        note(f"{stem}_touch.png",
             "the touch actually measured at that reference location, deepest-press "
             "frame (bottom row of step 1)",
             touch_index=int(idx), similarity_to_query=round(sim, 4),
             is_best_match=top1)
        refs.append(dict(rank=rank, touch_index=int(idx),
                         similarity=round(sim, 4), is_best_match=top1))
    man["step1_retrieval"] = dict(
        references_in_database=ret["n_db"], references_shown=len(refs),
        best_match_touch_index=int(ret["top1"]), shown=refs,
        similarity="DINOv3 ViT-B/16 cosine similarity between surface-normal renders")

    # --- step 2: the two wide renders and the correspondences ---------------
    # Two copies of each: with the red sensor-footprint box, as the figure draws
    # them, and without it, since the plain version is what the matcher ran on and
    # what the correspondence coordinates below refer to.
    # step 2 puts the query on the left and the reference it matched on the right
    for who, src, side in (("bestmatch", "match_left", "right"),
                           ("query", "match_right", "left")):
        raw = load_normal(f"{ASSETS}/{tag}_{src}.png")
        save(f"{out_dir}/step2_{who}_geometry_4x.png", sensor_box(raw))
        save(f"{out_dir}/step2_{who}_geometry_4x_nobox.png", raw)
        what = ("geometry of the best-matching reference" if who == "bestmatch"
                else "geometry of the query")
        note(f"step2_{who}_geometry_4x.png",
             f"{what} at 4x the sensor footprint, with the red box marking the sensor "
             f"itself; the {side} image of step 2")
        note(f"step2_{who}_geometry_4x_nobox.png",
             f"the same render without the red box -- this is what the feature matcher "
             f"actually ran on, and what step2_correspondences.csv is measured in")

    h, w = load(f"{ASSETS}/{tag}_match_left.png").shape[:2]
    with open(f"{out_dir}/step2_correspondences.csv", "w", newline="") as fh:
        c = csv.writer(fh)
        c.writerow(["left_x", "left_y", "right_x", "right_y", "agrees_with_warp"])
        for (ax, ay), (bx, by), ok in zip(md["xy_l"], md["xy_r"], md["inlier"]):
            c.writerow([f"{ax:.2f}", f"{ay:.2f}", f"{bx:.2f}", f"{by:.2f}", int(ok)])
    note("step2_correspondences.csv",
         "every SuperPoint + SuperGlue match between the two step-2 images, in pixels "
         "on those images, with a flag for the ones that agree on a single warp "
         "(the figure draws a sample of the agreeing ones as yellow lines)",
         image_width=w, image_height=h)

    np.savetxt(f"{out_dir}/step2_homography.txt", md["H"], fmt="%.10g")
    note("step2_homography.txt",
         "the 3x3 warp fitted to the agreeing correspondences, mapping best-match "
         "pixels to query pixels; this is what gets applied to the reference video")
    man["step2_alignment"] = dict(
        matches_proposed=int(len(md["xy_l"])),
        matches_agreeing_on_the_warp=int(md["inlier"].sum()),
        matcher="SuperPoint + SuperGlue on the surface-normal renders",
        warp="homography, fitted with RANSAC at an 8 pixel reprojection threshold")

    # --- step 3: what goes into the network and what comes out --------------
    pieces = [
        (f"{ASSETS}/{tag}_coarse_{mid - gap:03d}.png", f"step3_in_coarse_frame_{mid - gap:03d}.png",
         "the earlier of the two warped reference frames shown in step 3", load),
        (f"{ASSETS}/{tag}_coarse_{mid:03d}.png", f"step3_in_coarse_frame_{mid:03d}.png",
         "the warped reference video at the frame shown, one of the network's inputs", load),
        (f"{ASSETS}/{tag}_querynorm_scale100.png", "step3_in_query_normal_map.png",
         "the query location's normal map, stacked onto the network's input channels",
         load_normal),
        (f"{ASSETS}/{tag}_pred_{mid:03d}.png", "step3_out_refined_frame.png",
         "what the refinement network produces: the figure's final image", load),
    ]
    for src, name, what, reader in pieces:
        save(f"{out_dir}/{name}", reader(src))
        note(name, what)

    # the coarse transfer picture in step 2 is the same frame as one step-3 input
    shutil.copyfile(f"{out_dir}/step3_in_coarse_frame_{mid:03d}.png",
                    f"{out_dir}/step2_coarse_transfer.png")
    note("step2_coarse_transfer.png",
         "the coarse transfer shown at the right of step 2 -- the same picture as "
         f"step3_in_coarse_frame_{mid:03d}.png, repeated here for convenience")

    # --- extras that are not in the figure but usually wanted next to it -----
    if keep_raw:
        gt = f"{ASSETS}/{tag}_gt_{mid:03d}.png"
        if os.path.exists(gt):
            save(f"{out_dir}/extra_ground_truth_frame.png", load(gt))
            note("extra_ground_truth_frame.png",
                 "the touch that was really measured at the query location at this "
                 "frame. Not part of the method figure -- it is the answer the method "
                 "is trying to reproduce, useful if a reviewer asks")

    # --- whole-object renders, if render_object.py has been run into this folder
    rend = f"{out_dir}/object_renders"
    if os.path.isdir(rend):
        man["object_renders"] = dict(
            what="views of the whole textured mesh this touch was taken from "
                 f"(ObjectFolder object {md['obj']}), rendered by render_object.py",
            plain="object_renders/{obj}_view*.png - six views around the object",
            marked="object_renders/marked/{obj}_view*.png - four views with the touch "
                   "marked: blue is where the reference touch was taken, red is the "
                   "query location the figure predicts",
            closeup="object_renders/closeup/{obj}_view*.png - the same four views "
                    "framed tightly on the touch",
            sheets="each subfolder also has a {obj}_sheet.png contact sheet of its views")

    json.dump(man, open(f"{out_dir}/manifest.json", "w"), indent=1)
    with open(f"{out_dir}/README.txt", "w") as fh:
        fh.write(f"Individual pieces of the method figure for object "
                 f"{md['obj']}, touch {md['pair']}.\n\n"
                 "Every image here is real pipeline output and is saved exactly as it\n"
                 "appears in the figure: normal renders already have their empty\n"
                 "background repainted from black to white. manifest.json says what\n"
                 "each file is and carries the similarity numbers and match counts.\n\n"
                 "Regenerate with:\n"
                 f"  python paper_experiments/04_paper_figures/export_method_assets.py \\\n"
                 f"      --tag {tag} --ret_tag {ret['tag']} --mid {mid}\n"
                 + ("\nobject_renders/ holds views of the whole mesh this touch came\n"
                    "from; see manifest.json. Regenerate those with:\n"
                    f"  python paper_experiments/04_paper_figures/render_object.py \\\n"
                    f"      --obj {md['obj']} --views 6 --out <this folder>/object_renders\n"
                    if os.path.isdir(rend) else ""))
    print(f"{len(man['files'])} files written to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="984_6")
    ap.add_argument("--ret_tag", default="bench984_6")
    ap.add_argument("--n_db", type=int, default=4, help="reference touches shown")
    ap.add_argument("--mid", type=int, default=25, help="frame index used for stills")
    ap.add_argument("--frame_gap", type=int, default=1,
                    help="spacing of the two step-3 stills; match make_method.py")
    ap.add_argument("--out", default=None,
                    help="destination folder (default: method_assets_<tag>/ beside "
                         "the figures)")
    ap.add_argument("--no_extras", action="store_true",
                    help="write only what the figure itself shows")
    a = ap.parse_args()

    d = figure_inputs(a.tag, a.ret_tag, a.n_db, a.mid, a.frame_gap)
    export(d, a.out or f"{OUT}/method_assets_{a.tag}", not a.no_extras)


if __name__ == "__main__":
    main()
