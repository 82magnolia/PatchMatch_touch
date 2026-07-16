"""Post-processes an already-transferred video output directory (or a tree
of them, e.g. transfer_pipeline.py's per-session output), producing a new
directory with identical contents except the {idx}_transferred*.mp4 file(s),
which get masked against the query's render mask video -- i.e. adds
--use_mask-style compositing after the fact, without re-running the
(potentially expensive) correspondence + warp pipeline.

Mirrors main_retrieval_transfer_accel.py's/main_retrieval_transfer_feat_match.py's
--use_mask compositing exactly:
    output = mask * transferred + (1 - mask) * base_frame
where mask comes from {query_idx}_render_mask.mp4 in the ORIGINAL query_dir
(that mask video is never copied into transfer output directories, so it
must be looked up from the source data) and base_frame is frame 0 of
{query_idx}_ref_{video_type}.mp4, already present in the output directory.

Supports two output-directory shapes, auto-detected by walking --src_dir:
  flat:    <src_dir>/{idx}_transferred*.mp4
           (main_retrieval_transfer*.py --save_dir output)
  nested:  <src_dir>/<session>/transfer/{idx}_transferred*.mp4
           (transfer_pipeline.py output)
For the nested shape, pass --query_dir_root (e.g. log/real_data_gt_retrieval)
-- each output session subfolder name is looked up under that root, matching
transfer_pipeline.py's convention that a session's output subfolder name
equals its source session subfolder name.

Example usage (flat, single directory, sim data):
    python postprocess_mask_transfer.py \
        --src_dir log/transfer_feat_match \
        --query_dir Taxim/results/gen_contact_full_pseudo_mini \
        --out_dir log/transfer_feat_match_masked

Example usage (nested, transfer_pipeline.py's 100-session tree, real data):
    python postprocess_mask_transfer.py \
        --src_dir log/transfer_pipeline_real_data_gt_retrieval_sift_lightglue \
        --query_dir_root log/real_data_gt_retrieval \
        --out_dir log/transfer_pipeline_real_data_gt_retrieval_sift_lightglue_masked
"""

import argparse
import os
import re
import shutil
from os import path as osp

import cv2
import numpy as np

TRANSFERRED_RE = re.compile(r"^(\d+)_transferred.*\.mp4$")


def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame.astype(np.float32) / 255.0)
    cap.release()
    return frames, fps


def write_video(path, frames, fps):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()


def find_transfer_dirs(src_dir):
    """Every directory under src_dir containing >=1 {idx}_transferred*.mp4,
    including src_dir itself (the flat case)."""
    dirs = []
    for dirpath, _, filenames in os.walk(src_dir):
        if any(TRANSFERRED_RE.match(f) for f in filenames):
            dirs.append(dirpath)
    return sorted(dirs)


def resolve_query_dir(transfer_dir, src_dir, query_dir, query_dir_root):
    if query_dir is not None:
        return query_dir
    rel = osp.relpath(transfer_dir, src_dir)
    if rel == ".":
        return query_dir_root
    # nested case (transfer_pipeline.py): strip a trailing "transfer" component
    parts = rel.split(os.sep)
    if parts[-1] == "transfer":
        parts = parts[:-1]
    return osp.join(query_dir_root, *parts)


def mask_transferred_videos(transfer_dir, out_transfer_dir, query_dir, video_type):
    """Overwrites out_transfer_dir's (already-copied) {idx}_transferred*.mp4
    files in place with masked versions, reading source frames from the
    original transfer_dir. Leaves the file untouched (already a verbatim
    copy, from the earlier shutil.copytree) if the ref video or render mask
    can't be found."""
    for fname in sorted(os.listdir(transfer_dir)):
        m = TRANSFERRED_RE.match(fname)
        if not m:
            continue
        query_idx = int(m.group(1))
        src_path = osp.join(transfer_dir, fname)
        ref_path = osp.join(transfer_dir, f"{query_idx}_ref_{video_type}.mp4")
        mask_path = osp.join(query_dir, f"{query_idx}_render_mask.mp4")
        out_path = osp.join(out_transfer_dir, fname)

        if not osp.exists(ref_path):
            print(f"  [{fname}] Missing {ref_path} for base_frame; leaving unmasked.")
            continue
        if not osp.exists(mask_path):
            print(f"  [{fname}] Missing render mask {mask_path}; leaving unmasked.")
            continue

        transferred_frames, fps = read_video(src_path)
        ref_frames, _ = read_video(ref_path)
        mask_frames, _ = read_video(mask_path)
        if mask_frames[0].ndim == 3:
            mask_frames = [f.mean(axis=-1, keepdims=True) for f in mask_frames]
        base_frame = ref_frames[0]  # pre-contact background from reference

        masked = []
        for i, frame in enumerate(transferred_frames):
            mask = mask_frames[i] if i < len(mask_frames) else mask_frames[-1]
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))[..., None]
            masked.append(mask * frame + (1.0 - mask) * base_frame)
        write_video(out_path, masked, fps)
        print(f"  [{fname}] Masked -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Post-process already-transferred video(s) with the query's render mask.")
    parser.add_argument("--src_dir", required=True, type=str,
                        help="Existing transfer output directory (flat, or a tree containing "
                             "nested <session>/transfer/ subfolders, e.g. transfer_pipeline.py output).")
    parser.add_argument("--out_dir", required=True, type=str,
                        help="New directory to create; mirrors --src_dir's full structure, "
                             "with {idx}_transferred*.mp4 replaced by masked versions. Must not "
                             "already exist.")
    parser.add_argument("--query_dir", default=None, type=str,
                        help="Source query dir containing {idx}_render_mask.mp4 (flat --src_dir case).")
    parser.add_argument("--query_dir_root", default=None, type=str,
                        help="Root of per-session source query dirs (nested --src_dir case), e.g. "
                             "log/real_data_gt_retrieval -- each session subfolder name under "
                             "--src_dir must match one under this root.")
    parser.add_argument("--video_type", default="shadow", choices=["shadow", "sim"])
    args = parser.parse_args()
    if args.query_dir is None and args.query_dir_root is None:
        parser.error("One of --query_dir or --query_dir_root is required.")

    transfer_dirs = find_transfer_dirs(args.src_dir)
    if not transfer_dirs:
        print(f"No {{idx}}_transferred*.mp4 files found under: {args.src_dir}")
        return
    print(f"Found {len(transfer_dirs)} transfer output director(y/ies) under: {args.src_dir}")

    if osp.exists(args.out_dir):
        parser.error(f"--out_dir already exists: {args.out_dir} (refusing to overwrite/merge).")
    print(f"Mirroring {args.src_dir} -> {args.out_dir} ...")
    shutil.copytree(args.src_dir, args.out_dir)

    for transfer_dir in transfer_dirs:
        rel = osp.relpath(transfer_dir, args.src_dir)
        out_transfer_dir = osp.join(args.out_dir, rel) if rel != "." else args.out_dir
        query_dir = resolve_query_dir(transfer_dir, args.src_dir, args.query_dir, args.query_dir_root)
        print(f"\n=== {rel} (query_dir={query_dir}) ===")
        if not osp.isdir(query_dir):
            print(f"  Skipping: query_dir not found: {query_dir}")
            continue
        mask_transferred_videos(transfer_dir, out_transfer_dir, query_dir, args.video_type)

    print(f"\nDone. Masked output tree: {args.out_dir}")


if __name__ == "__main__":
    main()
