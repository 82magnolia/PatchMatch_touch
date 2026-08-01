"""Job 1 qualitative assets: the 3 x 6 ground-truth-retrieval figure, plus a
per-baseline comparison strip.

Main figure — one row per touch location, six columns:
    1. reference touch, middle frame          {idx}_ref_tactile_normal.mp4
    2. reference surface-normal render        gen_contact_full_.../{idx}_scale100_normal.jpg
    3. query surface-normal render            gen_contact_full_query_.../{idx}_scale100_normal.jpg
    4. query touch, coarse transfer           {idx}_transferred.mp4
    5. query touch, refined by the network    paper_job1_refine_ours/videos/{obj}_{idx}_enhanced.mp4
    6. ground-truth query touch               {idx}_query_tactile_normal.mp4

Columns 2 and 3 are the static geometry renders at the two sensor poses (the
benchmark's "normal rendering at the tactile sensor pose"), not video frames --
the videos in columns 1/4/5/6 are the tactile recordings, and "middle frame"
means the frame at the deepest press.

Every panel is written out individually as a PNG so the figure can be rebuilt or
re-cropped later without re-running anything, and a contact-sheet preview of the
whole matrix is written alongside.
"""
import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
REF_BASE = os.path.join(ROOT, "Taxim/results/gen_contact_full_tactile_normal_pseudo_mini")
QUERY_BASE = os.path.join(ROOT, "Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini")
# Coarse transfer built with the paper's default alignment (normals at 4x).
TRANSFER = os.path.join(ROOT, "log/paper_job1_transfer_normal")
REFINE = os.path.join(ROOT, "log/paper_job1_refine_ours_normal/videos")
BASELINES = {
    "quilting": os.path.join(ROOT, "log/paper_job1_baselines/quilting"),
    "inr": os.path.join(ROOT, "log/paper_job1_baselines/inr"),
}

COLUMNS = [
    ("ref_touch", "Reference touch"),
    ("ref_normal", "Reference normal"),
    ("query_normal", "Query normal"),
    ("coarse", "Coarse transfer (ours)"),
    ("refined", "Refined transfer (ours)"),
    ("gt", "Ground-truth query touch"),
]


def mid_frame(path):
    """Middle frame of a video as RGB, or None if unreadable."""
    if not os.path.exists(path):
        return None
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def read_jpg(path):
    if not os.path.exists(path):
        return None
    im = cv2.imread(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im is not None else None


def pick_rows(n_want):
    """Choose (object, touch) pairs that are both well predicted and worth looking at.

    Ranking purely by PSNR surfaces the flattest, most featureless contacts --
    they are easy to predict and show nothing. So among touches that beat the
    median PSNR, we pick the ones whose ground-truth frame has the most
    structure (highest pixel standard deviation), which is what a reader needs
    to see to judge the transfer.
    """
    mpath = os.path.join(ROOT, "log/paper_job1_refine_ours_normal/metrics.pkl")
    if not os.path.exists(mpath):
        return None
    with open(mpath, "rb") as f:
        m = pickle.load(f)
    # rebot_net/eval.py records per-object averages only, so score at object level
    # and then choose the most structured touch within each retained object.
    per_obj = m.get("per_object", {})
    if not per_obj:
        return None

    psnrs = sorted(v["PSNR"] for v in per_obj.values())
    median = psnrs[len(psnrs) // 2]
    good = [int(k) for k, v in per_obj.items() if v["PSNR"] >= median]

    scored = []
    for obj in good:
        for touch in range(8):
            gt = mid_frame(os.path.join(TRANSFER, str(obj),
                                        f"{touch}_query_tactile_normal.mp4"))
            if gt is None:
                continue
            if not os.path.exists(os.path.join(REFINE, f"{obj}_{touch}_enhanced.mp4")):
                continue
            scored.append((float(np.std(gt.astype(np.float32))), obj, touch))
    if not scored:
        return None
    scored.sort(reverse=True)

    # Spread the picks across distinct objects so the rows aren't near-duplicates.
    picks, seen = [], set()
    for _, obj, touch in scored:
        if obj in seen:
            continue
        seen.add(obj)
        picks.append((obj, touch))
        if len(picks) == n_want:
            break
    return picks


def label(img, text, height=22):
    """Stack a white caption strip above an image."""
    h, w = img.shape[:2]
    strip = np.full((height, w, 3), 255, np.uint8)
    cv2.putText(strip, text[:34], (3, height - 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([strip, img])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objects", nargs="+", type=int, default=None,
                   help="Object ids for the three rows (default: chosen by PSNR spread)")
    p.add_argument("--touch", type=int, default=None,
                   help="Touch index to show (default: the object's best touch)")
    p.add_argument("--out", default=os.path.join(ROOT, "log/paper_job1_figure_assets"))
    args = p.parse_args()

    if args.objects:
        pairs = [(o, args.touch if args.touch is not None else 0) for o in args.objects]
    else:
        pairs = pick_rows(3) or [(951, 0), (960, 0), (970, 0)]

    os.makedirs(args.out, exist_ok=True)
    manifest = {"columns": [c for c, _ in COLUMNS], "rows": []}
    rows = []

    for obj, touch in pairs:
        panels = {
            "ref_touch": mid_frame(os.path.join(TRANSFER, str(obj), f"{touch}_ref_tactile_normal.mp4")),
            "ref_normal": read_jpg(os.path.join(REF_BASE, str(obj), f"{touch}_scale100_normal.jpg")),
            "query_normal": read_jpg(os.path.join(QUERY_BASE, str(obj), f"{touch}_scale100_normal.jpg")),
            "coarse": mid_frame(os.path.join(TRANSFER, str(obj), f"{touch}_transferred.mp4")),
            "refined": mid_frame(os.path.join(REFINE, f"{obj}_{touch}_enhanced.mp4")),
            "gt": mid_frame(os.path.join(TRANSFER, str(obj), f"{touch}_query_tactile_normal.mp4")),
        }
        missing = [k for k, v in panels.items() if v is None]
        if missing:
            print(f"  [warn] object {obj} touch {touch}: missing {missing}")

        row_imgs = []
        for key, cap in COLUMNS:
            img = panels[key]
            if img is None:
                img = np.full((240, 320, 3), 240, np.uint8)
                cv2.putText(img, "N/A", (120, 125), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (90, 90, 90), 2, cv2.LINE_AA)
            out_path = os.path.join(args.out, f"obj{obj}_touch{touch}_{key}.png")
            cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            row_imgs.append(label(cv2.resize(img, (320, 240)), cap))

        # Baseline predictions for the same touch, saved next to the main panels.
        for meth, base in BASELINES.items():
            bp = os.path.join(base, str(obj), "transfer", f"{touch}_transferred.mp4")
            bimg = mid_frame(bp)
            if bimg is not None:
                cv2.imwrite(os.path.join(args.out, f"obj{obj}_touch{touch}_baseline_{meth}.png"),
                            cv2.cvtColor(bimg, cv2.COLOR_RGB2BGR))

        rows.append(np.hstack(row_imgs))
        manifest["rows"].append({"object": obj, "touch": touch,
                                 "missing": missing})

    sheet = np.vstack(rows)
    sheet_path = os.path.join(args.out, "contact_sheet.png")
    cv2.imwrite(sheet_path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"rows: {[(r['object'], r['touch']) for r in manifest['rows']]}")
    print(f"panels + contact sheet -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
