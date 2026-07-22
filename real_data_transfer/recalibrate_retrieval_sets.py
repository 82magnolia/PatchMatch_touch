"""Rebuild the retrieval datasets with the current ArUco->GelSight calibration.

Every per-touch artifact under log/real_data_{gt,real}_retrieval -- the .npz
normals/contact data and the static modality renderings -- was produced by
ortho_project_raw, which reads the ARUCO_TO_CONTACT_* constants in
_gelsight_processing.py. Those constants changed, so the artifacts are stale
while the raw session inputs (video, poses, diffs, object caches) are not.

This walks each session, hardlinks the session-level inputs into the output
tree, and re-runs process_single_shot.reprocess to regenerate the per-touch
files. Hardlinks keep the copy cheap: the inputs are ~35% of the bytes and are
never written to.

The reprocess parameters below are not guesses -- rerunning with the *old*
constants reproduces the committed .npz files bit-exactly, so calibration is the
only thing that differs in the output.

Usage:
  conda activate pm_real
  python real_data_transfer/recalibrate_retrieval_sets.py
  python real_data_transfer/recalibrate_retrieval_sets.py --dry_run
  python real_data_transfer/recalibrate_retrieval_sets.py --src log/real_data_gt_retrieval
"""

import argparse
import os
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _gelsight_processing import (
    ARUCO_TO_CONTACT_X_M, ARUCO_TO_CONTACT_Y_M,
    ARUCO_TO_CONTACT_M, ARUCO_TO_CONTACT_THETA_DEG,
)
from process_single_shot import reprocess

DEFAULT_SRCS = ["log/real_data_gt_retrieval", "log/real_data_real_retrieval"]

# Parameters verified to reproduce the existing .npz outputs bit-exactly under
# the old constants; do not change these without re-running that check. The
# {i}_render_mask.mp4 frames now differ from the committed ones: reprocess
# aligns the mask to the shadow contact window by default (see
# render_mask_eval_positions), which the old outputs predate.
REPROCESS_KWARGS = dict(
    seg_threshold=None,          # from each session_meta.json
    min_gap_frames=10,
    peak_ratio=0.01,
    num_frames=None,             # -> 50
    render_mask_type="hard",
    mask_temperature=0.002,
    render_mask_thres=-0.005,
    inpaint_method="telea",
    render_scale_list=[1.0, 2.0, 4.0, 8.0],
    dry_run=False,
    merge_gap=0,
    boundary_pad=10,
    unmasked=False,
    tactile_normal_video=True,
    render_mask_depth_mode="aruco",
)

# Per-touch artifacts are regenerated; everything else is a session input.
TOUCH_FILE_RE = re.compile(r"^\d+_")


def link_inputs(src_dir, dst_dir):
    """Hardlink session-level inputs into dst_dir. Falls back to copy across
    filesystems. Returns the number of files linked."""
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(src_dir)):
        if TOUCH_FILE_RE.match(name):
            continue
        s, d = os.path.join(src_dir, name), os.path.join(dst_dir, name)
        if not os.path.isfile(s) or os.path.exists(d):
            continue
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)
        n += 1
    return n


def session_dirs(src):
    """Numeric session subdirectories, in numeric order."""
    return sorted((d for d in os.listdir(src)
                   if d.isdigit() and os.path.isdir(os.path.join(src, d))),
                  key=int)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", nargs="+", default=DEFAULT_SRCS,
                   help="Source dataset roots (default: both retrieval sets)")
    p.add_argument("--suffix", default="_recalib",
                   help="Appended to each source root to form the output root")
    p.add_argument("--dry_run", action="store_true",
                   help="List what would be rebuilt without writing files")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip sessions whose output dir already has 0_meta.json, "
                        "so an interrupted run can be resumed")
    args = p.parse_args()

    print("Rebuilding with calibration: "
          f"X={ARUCO_TO_CONTACT_X_M*1e3:.2f}mm  Y={ARUCO_TO_CONTACT_Y_M*1e3:.2f}mm  "
          f"Z={ARUCO_TO_CONTACT_M*1e3:.2f}mm  Theta={ARUCO_TO_CONTACT_THETA_DEG:.2f}deg\n")

    for src in args.src:
        if not os.path.isdir(src):
            sys.exit(f"No such dataset root: {src}")
        dst_root = src.rstrip("/") + args.suffix
        sessions = session_dirs(src)
        print(f"{src} -> {dst_root}   ({len(sessions)} sessions)")

        if args.dry_run:
            print(f"  would rebuild sessions: {', '.join(sessions[:8])}"
                  f"{' ...' if len(sessions) > 8 else ''}\n")
            continue

        t_root = time.time()
        done = skipped = failed = 0
        for i, sess in enumerate(sessions, 1):
            s_dir = os.path.join(src, sess)
            d_dir = os.path.join(dst_root, sess)

            if args.skip_existing and os.path.exists(os.path.join(d_dir, "0_meta.json")):
                skipped += 1
                continue

            print(f"  [{i}/{len(sessions)}] {sess}", flush=True)
            link_inputs(s_dir, d_dir)
            t0 = time.time()
            try:
                reprocess(session_dir=s_dir, output_dir=d_dir, **REPROCESS_KWARGS)
            except SystemExit as exc:
                # reprocess calls sys.exit on missing inputs; one bad session
                # must not abandon the other 99.
                print(f"    SKIPPED: {exc}")
                failed += 1
                continue
            done += 1
            print(f"    {time.time() - t0:.1f}s", flush=True)

        print(f"  {done} rebuilt, {skipped} skipped, {failed} failed "
              f"in {(time.time() - t_root)/60:.1f} min\n")


if __name__ == "__main__":
    main()
