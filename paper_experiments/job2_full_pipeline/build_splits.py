"""Build the query / reference split for the full-pipeline benchmark.

The full-pipeline benchmark (Taxim/results/gen_contact_raw_eval_tactile_normal_
pseudo_mini) stores every touch of an object in one flat directory. The pipeline
under test takes a *reference set* and a *query set* in separate directories, so
this script materialises a per-object split as two symlink directories:

    log/paper_job2_bench/{obj}/ref/     the touches the pipeline may retrieve from
    log/paper_job2_bench/{obj}/query/   the held-out touches to predict

Selection rule (experiment_plan: "select 3~5 query touch per object, rest as
reference"): 4 held-out queries per object, dropped to 3 for the handful of
objects with fewer than 9 touches so at least a few references remain. The
choice is a seeded random sample (seed 0), so the split is reproducible.

Original touch indices are preserved in the symlink names. That matters: the
refinement network reads its query normal render as
{cond_dir}/{obj}/{idx}_scale100_normal.jpg, so the predicted video's index must
still identify the original touch.
"""
import json
import os
import random
import re
import sys

import cv2

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
SRC = os.path.join(ROOT, "Taxim/results/gen_contact_raw_eval_tactile_normal_pseudo_mini")
OUT = os.path.join(ROOT, "log/paper_job2_bench")
SEED = 0


def touch_indices(obj_dir):
    """Readable touch indices of an object.

    A handful of touches in the benchmark were written truncated (44-byte files
    with no moov atom, so OpenCV reports zero frames). They are dropped here
    rather than downstream, where an empty frame list crashes the transfer's
    video writer. The dropped touches are reported in the manifest.
    """
    idxs, broken = [], []
    for name in os.listdir(obj_dir):
        m = re.match(r"^(\d+)_tactile_normal\.mp4$", name)
        if not m:
            continue
        idx = int(m.group(1))
        cap = cv2.VideoCapture(os.path.join(obj_dir, name))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        (idxs if n_frames > 0 else broken).append(idx)
    return sorted(idxs), sorted(broken)


def link_touch(src_dir, dst_dir, idx):
    """Symlink every per-touch file of `idx` into dst_dir, keeping its name.

    Matches on the full leading integer so touch 1 never picks up touch 10's files.
    """
    n = 0
    for name in os.listdir(src_dir):
        m = re.match(r"^(\d+)_", name)
        if m is None or int(m.group(1)) != idx:
            continue
        dst = os.path.join(dst_dir, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(src_dir, name), dst)
        n += 1
    return n


def main():
    objs = sorted(os.listdir(SRC), key=int)
    manifest = {}
    os.makedirs(OUT, exist_ok=True)

    for obj in objs:
        src_dir = os.path.join(SRC, obj)
        idxs, broken = touch_indices(src_dir)
        if len(idxs) < 5:
            print(f"  [skip] object {obj}: only {len(idxs)} touches")
            continue

        n_query = 4 if len(idxs) >= 9 else 3
        rng = random.Random(f"{SEED}-{obj}")
        query_idx = sorted(rng.sample(idxs, n_query))
        ref_idx = [i for i in idxs if i not in query_idx]

        ref_dir = os.path.join(OUT, obj, "ref")
        query_dir = os.path.join(OUT, obj, "query")
        os.makedirs(ref_dir, exist_ok=True)
        os.makedirs(query_dir, exist_ok=True)
        for i in ref_idx:
            link_touch(src_dir, ref_dir, i)
        for i in query_idx:
            link_touch(src_dir, query_dir, i)

        manifest[obj] = {"n_touches": len(idxs), "query": query_idx,
                         "ref": ref_idx, "dropped_unreadable": broken}

    path = os.path.join(ROOT, "paper_experiments/job2_full_pipeline/splits.json")
    with open(path, "w") as f:
        json.dump({"seed": SEED, "source": SRC, "objects": manifest}, f, indent=2)

    n_q = sum(len(v["query"]) for v in manifest.values())
    n_r = sum(len(v["ref"]) for v in manifest.values())
    print(f"{len(manifest)} objects | {n_q} query touches | {n_r} reference touches")
    print(f"manifest -> {path}")


if __name__ == "__main__":
    sys.exit(main())
