"""Score every evaluation touch for how well it would work as a paper figure.

PSNR alone is a bad selector: the highest-PSNR touches are usually the ones where
almost nothing happens (a faint contact on a flat patch), which look empty in a
figure. This script scores each of the 400 ground-truth-retrieval evaluation
touches on the properties that actually make a good qualitative example, and
writes the result to candidates.json for the figure builders to consume.

Criteria computed per touch
  contact         how strongly the ground-truth touch deviates from flat gel
                  (a visible contact, not a whisper)
  structure       edge density inside the contact region (interesting geometry,
                  not a single smooth blob)
  refined_psnr    prediction quality after the refinement network
  gain            refined PSNR minus coarse PSNR -- how much visible work the
                  refinement stage is doing in this example
  pose_diff       how different the query sensor pose is from the reference
                  (a non-trivial analogy rather than a near-identical re-touch)
  render_cover    fraction of the 4x normal render that actually contains
                  surface rather than empty background (so columns 2-3 are
                  informative and the object is recognisable)
  temporal        how much the contact grows and shrinks over the press
                  (only matters for the reconstruction figure, whose columns
                  are frames)
"""
import json
import os
import pickle

import cv2
import numpy as np

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
JOB2 = f"{ROOT}/log/paper_job02_gt_retrieval_figure_normalmatch"
TRANSFER = f"{ROOT}/log/transfer_feat_match_pseudo_mini_tactile_normal_superpoint_superglue_normalmatch"
REF_RENDER = f"{ROOT}/Taxim/results/gen_contact_full_tactile_normal_pseudo_mini"
QUERY_RENDER = f"{ROOT}/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candidates.json")

FLAT = np.array([0.0, 0.0, 1.0])


def dev_from_flat(rgb01):
    """Per-pixel distance of the encoded normal from the flat-gel normal (0,0,1)."""
    return np.linalg.norm(2.0 * rgb01 - 1.0 - FLAT, axis=-1)


def load01(path):
    im = cv2.imread(path)
    if im is None:
        return None
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def video_dev_profile(path):
    """Mean deviation-from-flat per frame, over a whole touch video."""
    cap = cv2.VideoCapture(path)
    prof = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        prof.append(float(dev_from_flat(rgb).mean()))
    cap.release()
    return np.array(prof)


def render_cover(path):
    """Fraction of a normal render that is actual surface, not empty background.

    Background pixels are rendered black (all three channels near zero), which is
    never a valid encoded normal, so a simple brightness test separates them.
    """
    im = cv2.imread(path)
    if im is None:
        return 0.0
    return float((im.max(axis=2) > 25).mean())


def main():
    recs = pickle.load(open(f"{JOB2}/per_touch_metrics.pkl", "rb"))
    out = []
    for i, r in enumerate(recs):
        obj, pair, ridx = r["obj"], r["pair"], r["ref_idx"]
        a = f"{JOB2}/assets/{obj}_{pair}"

        gt = load01(f"{a}_06_gt_query.png")
        ref = load01(f"{a}_01_ref_touch.png")
        if gt is None or ref is None:
            continue

        d_gt = dev_from_flat(gt)
        contact = float(d_gt.mean())
        mask = d_gt > max(0.15, 0.35 * d_gt.max())

        # Edge density inside the contact region: distinguishes a shaped contact
        # (an edge, a rim, a tip) from one broad smooth bulge.
        g = (np.clip(gt, 0, 1) * 255).astype(np.uint8)
        edges = cv2.Canny(cv2.cvtColor(g, cv2.COLOR_RGB2GRAY), 60, 160) > 0
        structure = float(edges[mask].mean()) if mask.sum() > 50 else 0.0

        # How different the query pose is from the reference pose. Both mid
        # frames live in their own sensor frames, so a plain difference of the
        # two tactile images is a fair proxy for "the analogy is non-trivial".
        pose_diff = float(np.abs(gt - ref).mean())

        cover_r = render_cover(f"{REF_RENDER}/{obj}/{ridx}_scale25_normal.jpg")
        cover_q = render_cover(f"{QUERY_RENDER}/{obj}/{pair}_scale25_normal.jpg")

        prof = video_dev_profile(f"{TRANSFER}/{obj}/{pair}_query_tactile_normal.mp4")
        if prof.size:
            temporal = float((prof.max() - prof.min()) / (prof.max() + 1e-8))
        else:
            temporal = 0.0

        out.append(dict(
            obj=obj, pair=pair, ref_idx=ridx,
            coarse_psnr=r["coarse"]["PSNR"], refined_psnr=r["refined"]["PSNR"],
            refined_ssim=r["refined"]["SSIM"], refined_lpips=r["refined"]["LPIPS"],
            gain=r["refined"]["PSNR"] - r["coarse"]["PSNR"],
            contact=contact, structure=structure, pose_diff=pose_diff,
            cover_ref=cover_r, cover_query=cover_q,
            temporal=temporal, n_frames=r["n_frames"], mid_frame=r["mid_frame"],
        ))
        if (i + 1) % 50 == 0:
            print(f"  scored {i+1}/{len(recs)}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT}  ({len(out)} touches)")


if __name__ == "__main__":
    main()
