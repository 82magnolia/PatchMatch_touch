"""Job 4: runtime of the full pipeline for N reference touches and 1 query.

The paper reports three numbers, each EXCLUDING a heavy generic feature-extraction
step so the cost of our own machinery is what is measured:

  1. "Retrieval phase after DINOv3 feature extraction" — ranking N reference
     touches against the query, given features are already extracted.
  2. "Coarse alignment after local feature matching" — RANSAC linear fit, offset
     estimation, dense-field construction and per-frame warping, given the
     sparse matches are already computed.
  3. "Neural network-based refinement" — per output frame.

Stages 1 and 2 are timed by wrapping the excluded functions (extract_features,
sparse_match) with a counter and subtracting their accumulated time from the
stage total, so no code in the pipeline itself has to change.

All timings are taken on one RTX 3090 with CUDA synchronised around each stage,
after a warm-up pass that pays one-off model-loading and cuDNN autotune costs.
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "rebot_net"))


class Accum:
    """Wraps a function to accumulate its wall-clock cost across calls."""

    def __init__(self):
        self.total = 0.0
        self.calls = 0

    def wrap(self, fn):
        def inner(*a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                self.total += time.perf_counter() - t0
                self.calls += 1
        return inner

    def reset(self):
        self.total = 0.0
        self.calls = 0


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_retrieval(query_dir, ref_dir, dino_weights, device, repeats):
    """Stage 1: DINOv3 retrieval, split into extraction and ranking."""
    import retrieve_touch as rt

    model, transform = rt.load_dino_model("dinov3_vitb16", dino_weights, device)

    ref_files = rt.discover_files(ref_dir, "normal", 100)
    query_files = rt.discover_files(query_dir, "normal", 100)
    ref_paths = [p for _, p in ref_files]
    query_paths = [p for _, p in query_files][:1]      # one query touch
    n_ref = len(ref_paths)

    extract = Accum()
    orig_extract = rt.extract_features
    rt.extract_features = extract.wrap(orig_extract)
    try:
        rank_times, total_times = [], []
        for i in range(repeats + 1):
            extract.reset()
            sync()
            t0 = time.perf_counter()
            ref_feats = rt.extract_features(model, transform, ref_paths, device)
            q_feats = rt.extract_features(model, transform, query_paths, device)
            rt.compute_topk(q_feats, ref_feats, k=5)
            sync()
            total = time.perf_counter() - t0
            if i == 0:
                continue                                # warm-up
            total_times.append(total)
            rank_times.append(total - extract.total)
    finally:
        rt.extract_features = orig_extract

    return {
        "n_reference_touches": n_ref,
        "extraction_plus_ranking_s": statistics.mean(total_times),
        "ranking_after_extraction_s": statistics.mean(rank_times),
        "ranking_after_extraction_std_s": statistics.pstdev(rank_times) if len(rank_times) > 1 else 0.0,
    }


def time_coarse_alignment(query_dir, ref_dir, query_idx, ref_idx, repeats):
    """Stage 2: coarse alignment, excluding the sparse feature-matching calls."""
    import main_retrieval_transfer_feat_match as fm
    import decomposed_match as dm

    match = Accum()
    orig_sparse = dm.sparse_match
    dm.sparse_match = match.wrap(orig_sparse)

    n_frames = None
    try:
        align_times, warp_times, totals = [], [], []
        for i in range(repeats + 1):
            match.reset()
            sync()
            t0 = time.perf_counter()
            nnf, info = fm.compute_transfer_nnf(
                query_dir, ref_dir, query_idx, ref_idx, ["curvature"],
                video_scale=100.0, match_scale=25.0, convention="obj_scale_factor",
                transform_type="homography", reproj_threshold=8.0,
                linear_matcher="superpoint_superglue",
                offset_matcher="superpoint_superglue",
                offset_method="median")
            sync()
            t_nnf = time.perf_counter() - t0
            nnf_excl = t_nnf - match.total

            ref_frames, _fps = fm.read_video(
                os.path.join(ref_dir, f"{ref_idx}_tactile_normal.mp4"))
            n_frames = len(ref_frames)
            sync()
            t1 = time.perf_counter()
            for frame in ref_frames:
                fm.reconstruct_avg(nnf, frame, patch_size=1)
            sync()
            t_warp = time.perf_counter() - t1

            if i == 0:
                continue                                # warm-up
            align_times.append(nnf_excl)
            warp_times.append(t_warp)
            totals.append(t_nnf + t_warp)
    finally:
        dm.sparse_match = orig_sparse

    mean_align = statistics.mean(align_times)
    mean_warp = statistics.mean(warp_times)
    return {
        "n_frames": n_frames,
        "fit_after_matching_s": mean_align,
        "frame_warping_s": mean_warp,
        "coarse_after_matching_s": mean_align + mean_warp,
        "coarse_after_matching_std_s": statistics.pstdev(
            [a + w for a, w in zip(align_times, warp_times)]) if len(align_times) > 1 else 0.0,
        "including_matching_s": statistics.mean(totals),
    }


def time_refinement(checkpoint, device, n_frames, repeats, h=240, w=320):
    """Stage 3: ReBotNet refinement forward pass, per frame."""
    from train import build_model

    # geom_concat normal + temporal FiLM -> 6 cond channels (3 per frame), FiLM off.
    model = build_model("rebot_S", cond_chans=3, film_chans=0,
                        bottleneck_hw=24, time_cond="film").to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    lq = torch.rand(1, 2, 6, h, w, device=device)
    t = torch.tensor([0.5], device=device)

    with torch.no_grad():
        for _ in range(3):                              # warm-up
            model(lq, film=None, t=t)
        sync()
        per_frame = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(n_frames):
                model(lq, film=None, t=t)
            sync()
            per_frame.append((time.perf_counter() - t0) / n_frames)

    return {
        "frames_timed": n_frames,
        "refinement_per_frame_s": statistics.mean(per_frame),
        "refinement_per_frame_std_s": statistics.pstdev(per_frame) if len(per_frame) > 1 else 0.0,
        "refinement_per_video_s": statistics.mean(per_frame) * n_frames,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", default="1", help="Object id in the full-pipeline benchmark")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out", default=os.path.join(
        ROOT, "paper_experiments/job4_runtime/runtime.json"))
    args = p.parse_args()

    bench = os.path.join(ROOT, "log/paper_job2_bench", args.object)
    ref_dir = os.path.join(bench, "ref")
    query_dir = os.path.join(bench, "query")
    dino = os.path.join(ROOT, "dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    ckpt = os.path.join(ROOT, "log/rebot_checkpoints_S_geomcat_film/best.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"Object {args.object} | device {gpu} | {args.repeats} repeats\n")

    print("[1/3] retrieval ...")
    retrieval = time_retrieval(query_dir, ref_dir, dino, device, args.repeats)
    print(json.dumps(retrieval, indent=2))

    # Use the pipeline's own retrieval result so the timed pair is the real one.
    import re
    q_idx = sorted(int(re.match(r"^(\d+)_", f).group(1))
                   for f in os.listdir(query_dir) if f.endswith("_tactile_normal.mp4"))[0]
    r_idx = sorted(int(re.match(r"^(\d+)_", f).group(1))
                   for f in os.listdir(ref_dir) if f.endswith("_tactile_normal.mp4"))[0]

    print("\n[2/3] coarse alignment ...")
    coarse = time_coarse_alignment(query_dir, ref_dir, q_idx, r_idx, args.repeats)
    print(json.dumps(coarse, indent=2))

    print("\n[3/3] refinement ...")
    refine = time_refinement(ckpt, device, coarse["n_frames"] or 50, args.repeats)
    print(json.dumps(refine, indent=2))

    out = {
        "gpu": gpu,
        "object": args.object,
        "query_idx": q_idx,
        "ref_idx": r_idx,
        "repeats": args.repeats,
        "retrieval": retrieval,
        "coarse_alignment": coarse,
        "refinement": refine,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 62)
    print(f"N = {retrieval['n_reference_touches']} reference touches, 1 query, "
          f"{coarse['n_frames']} frames per touch video")
    print(f"  retrieval after DINOv3 feature extraction : "
          f"{retrieval['ranking_after_extraction_s'] * 1000:.2f} ms")
    print(f"  coarse alignment after feature matching   : "
          f"{coarse['coarse_after_matching_s']:.3f} s")
    print(f"  network refinement                        : "
          f"{refine['refinement_per_frame_s'] * 1000:.1f} ms / frame")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
