"""Find touches that make good method-figure examples.

The method figure tells one story with one touch: a query location is matched
against that object's reference touches, the winning reference is lined up with
the query, and the warped video is refined. For that story to read well the
touch has to satisfy several things at once, which this script measures over
every query of the evaluation objects (951-1000):

  1. retrieval is correct   the top-1 reference is the ground-truth pairing, so
                            the "best match" label in the figure is honest
  2. the winner stands out  a large gap between the top-1 and top-2 similarity,
                            so the bar chart in step 1 is readable
  3. the weaker references look different, so the row of four thumbnails is not
     four near-copies
  4. the geometry has structure (not a blank patch), measured as how much of the
     surface normal render deviates from a flat facing-up surface
  5. plenty of SuperPoint + SuperGlue inliers, so step 2 can draw many lines

Stage "ret" does 1-4 with DINOv3 (cheap). Stage "match" runs the actual feature
matcher on the shortlist and counts inliers.
"""
import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
import torch

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
sys.path.insert(0, ROOT)

OUT = f"{ROOT}/log/paper_job04_paper_figures"
QUERY_RENDER = f"{ROOT}/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini"
REF_RENDER = f"{ROOT}/Taxim/results/gen_contact_full_tactile_normal_pseudo_mini"
TRANSFER = f"{ROOT}/log/transfer_feat_match_pseudo_mini_tactile_normal_superpoint_superglue_normalmatch"
DINO_W = f"{ROOT}/dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"


def structure_score(path):
    """How much surface relief the render shows, in [0, 1].

    A flat patch renders as a near-constant colour; a ridge or a hole makes the
    surface normals swing around. We take the fraction of pixels whose colour is
    far from the image's own most common colour.
    """
    im = cv2.imread(path).astype(np.float32) / 255.0
    flat = np.median(im.reshape(-1, 3), axis=0)
    d = np.linalg.norm(im - flat, axis=2)
    return float((d > 0.15).mean())


def scan_retrieval(objs, out_json):
    from retrieve_touch import load_dino_model, extract_features

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = load_dino_model("dinov3_vitb16", DINO_W, device)

    rows = []
    for oi, obj in enumerate(objs):
        rdir, qdir = f"{REF_RENDER}/{obj}", f"{QUERY_RENDER}/{obj}"
        if not (os.path.isdir(rdir) and os.path.isdir(qdir)):
            continue
        idxs = sorted(int(f.split("_")[0]) for f in os.listdir(rdir)
                      if f.endswith("_scale100_normal.jpg"))
        rpaths = [f"{rdir}/{i}_scale100_normal.jpg" for i in idxs]
        qpaths = [f"{qdir}/{i}_scale100_normal.jpg" for i in idxs
                  if os.path.exists(f"{qdir}/{i}_scale100_normal.jpg")]
        if not qpaths or len(idxs) < 4:
            continue
        rf = extract_features(model, transform, rpaths, device)
        qf = extract_features(model, transform, qpaths, device)
        sims = (qf @ rf.T).numpy()
        rr = (rf @ rf.T).numpy()

        for qi, q in enumerate(int(os.path.basename(p).split("_")[0]) for p in qpaths):
            s = sims[qi]
            order = np.argsort(-s)
            top1, top2 = idxs[order[0]], idxs[order[1]]
            j = idxs.index(q)
            # how alike the reference database is to the winner: low means the
            # other thumbnails in the figure will look clearly different
            others = [rr[order[0], k] for k in range(len(idxs)) if k != order[0]]
            rows.append(dict(
                obj=obj, query=q, top1=top1, correct=bool(top1 == q),
                s1=float(s[order[0]]), s2=float(s[order[1]]),
                margin=float(s[order[0]] - s[order[1]]), top2=top2,
                spread=float(np.max(s) - np.min(s)),
                db_similarity=float(np.mean(others)),
                structure=structure_score(f"{qdir}/{q}_scale100_normal.jpg"),
                has_video=os.path.exists(f"{TRANSFER}/{obj}/{q}_ref_tactile_normal.mp4"),
            ))
        print(f"[{oi + 1}/{len(objs)}] object {obj}: {len(qpaths)} queries", flush=True)

    json.dump(rows, open(out_json, "w"), indent=1)
    print("wrote", out_json, len(rows), "queries")
    return rows


def scan_matches(rows, out_json, top_n):
    """Count SuperPoint + SuperGlue inliers for the shortlist."""
    sys.path.insert(0, ROOT)
    from imcui_match import compute_imcui_sparse_matches

    for k, r in enumerate(rows[:top_n]):
        obj, q, ref = r["obj"], r["query"], r["top1"]
        left = cv2.cvtColor(cv2.imread(f"{REF_RENDER}/{obj}/{ref}_scale25_normal.jpg"),
                            cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        right = cv2.cvtColor(cv2.imread(f"{QUERY_RENDER}/{obj}/{q}_scale25_normal.jpg"),
                             cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        try:
            pl, pr = compute_imcui_sparse_matches(left, right, "superpoint_superglue")
            xy_l, xy_r = pl[:, ::-1].astype(np.float32), pr[:, ::-1].astype(np.float32)
            H, mask = cv2.findHomography(xy_l, xy_r, cv2.RANSAC, 8.0)
            n_in = int(mask.ravel().sum()) if mask is not None else 0
            # a good picture spreads its lines over the whole patch
            span = float(np.std(xy_r[mask.ravel().astype(bool)], axis=0).mean()) \
                if n_in > 2 else 0.0
        except Exception as e:                       # noqa: BLE001
            print("  match failed:", e)
            n_in, len_all, span = 0, 0, 0.0
            r.update(n_matches=0, n_inliers=0, inlier_span=0.0)
            continue
        r.update(n_matches=int(len(xy_l)), n_inliers=n_in, inlier_span=span)
        print(f"[{k + 1}/{top_n}] object {obj} query {q} -> ref {ref}: "
              f"{n_in}/{len(xy_l)} inliers, spread {span:.1f}px", flush=True)

    json.dump(rows[:top_n], open(out_json, "w"), indent=1)
    print("wrote", out_json)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", nargs="+", default=["ret", "match"])
    p.add_argument("--objs", nargs="+", type=int,
                   default=list(range(951, 1001)))
    p.add_argument("--top_n", type=int, default=30)
    a = p.parse_args()

    ret_json = f"{OUT}/method_candidates_retrieval.json"
    if "ret" in a.stage:
        rows = scan_retrieval(a.objs, ret_json)
    else:
        rows = json.load(open(ret_json))

    keep = [r for r in rows if r["correct"] and r["has_video"]
            and r["structure"] > 0.15 and r["margin"] > 0.08]
    keep.sort(key=lambda r: -(r["margin"] + 0.5 * r["s1"] + 0.5 * r["structure"]
                              - 0.5 * r["db_similarity"]))
    print(f"\n{len(keep)} of {len(rows)} queries pass the filters")
    if "match" in a.stage:
        scan_matches(keep, f"{OUT}/method_candidates.json", a.top_n)


if __name__ == "__main__":
    main()
