"""Cache every real image / number the teaser and method figures need.

Nothing in the two figure scripts is drawn from imagination: the tactile
frames come out of the benchmark videos, the predicted frames come out of the
actual refinement network, the feature-match lines come out of an actual
SuperPoint + SuperGlue run, and the retrieval similarity bars come out of an
actual DINOv3 comparison. This script produces all of that once and writes it
under log/paper_job04_paper_figures/assets/ so the layout scripts stay fast
and can be re-run many times while iterating on the design.

Three jobs, each skippable:

  --do touch      one touch from the ground-truth-retrieval benchmark:
                  reference / coarse-transfer / refined / ground-truth frames
                  plus the two surface normal renders
  --do match      SuperPoint + SuperGlue correspondences between the reference
                  and query normal renders of that same touch
  --do retrieval  DINOv3 leave-one-out retrieval over the 10 touches of one
                  object from the full-pipeline benchmark
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np
import torch

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/rebot_net")

OUT = f"{ROOT}/log/paper_job04_paper_figures"
ASSETS = f"{OUT}/assets"

# The paper default: normal-modality matching at 4x the sensor footprint
# (see paper_experiments/02_gt_retrieval_figure/results.md).
TRANSFER = f"{ROOT}/log/transfer_feat_match_pseudo_mini_tactile_normal_superpoint_superglue_normalmatch"
QUERY_RENDER = f"{ROOT}/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini"
REF_RENDER = f"{ROOT}/Taxim/results/gen_contact_full_tactile_normal_pseudo_mini"
EVAL_RENDER = f"{ROOT}/Taxim/results/gen_contact_raw_eval_tactile_normal_pseudo_mini"
CKPT = f"{ROOT}/log/rebot_checkpoints_S_geomcat_film/best.pth"
DINO_W = f"{ROOT}/dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"


def save_png(path, rgb01):
    cv2.imwrite(path, cv2.cvtColor((np.clip(rgb01, 0, 1) * 255).round().astype(np.uint8),
                                   cv2.COLOR_RGB2BGR))


def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    cap.release()
    return frames


# --------------------------------------------------------------------- touch
def do_touch(obj, pair, tag):
    """Reference / coarse / refined / ground-truth frames for one touch."""
    from dataset import TactileTransferDataset
    from train import build_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # geomcat_film: query normal render concatenated as 3 extra input channels,
    # timestamp injected through FiLM. Same configuration as job 2's evaluation.
    model = build_model("rebot_S", cond_chans=3, film_chans=0,
                        bottleneck_hw=24, time_cond="film").to(device)
    ck = torch.load(CKPT, map_location=device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    print(f"refinement network loaded, epoch {ck.get('epoch')}")

    ds = TactileTransferDataset(TRANSFER, [obj], split="test", cond_dir=QUERY_RENDER,
                                film_modality="normal", film_scale=100,
                                geom_concat=True, video_type="tactile_normal",
                                time_cond="film")
    preds, gts, coarses = [], [], []
    with torch.no_grad():
        for lq, gt, blank, film, t_norm in ds.iter_video_pairs(obj, pair):
            t_in = torch.tensor([t_norm], device=device)
            pr = model(lq.unsqueeze(0).to(device), film=None, t=t_in).squeeze(0)
            preds.append(pr.cpu().clamp(0, 1).permute(1, 2, 0).numpy())
            gts.append(gt.permute(1, 2, 0).numpy())
            coarses.append(lq[1, :3].permute(1, 2, 0).numpy())   # current coarse frame

    refs = read_video(f"{TRANSFER}/{obj}/{pair}_ref_tactile_normal.mp4")
    n = len(preds)
    print(f"object {obj} touch {pair}: {n} refined frames, {len(refs)} reference frames")

    for i in range(n):
        save_png(f"{ASSETS}/{tag}_pred_{i:03d}.png", preds[i])
        save_png(f"{ASSETS}/{tag}_gt_{i:03d}.png", gts[i])
        save_png(f"{ASSETS}/{tag}_coarse_{i:03d}.png", coarses[i])
    for i, fr in enumerate(refs):
        save_png(f"{ASSETS}/{tag}_ref_{i:03d}.png", fr)

    # The two geometry renders, at 1x the sensor footprint and at the 4x field
    # of view the matching actually runs on.
    for who, root, idx in (("refnorm", REF_RENDER, pair), ("querynorm", QUERY_RENDER, pair)):
        for sc in ("100", "25"):
            src = f"{root}/{obj}/{idx}_scale{sc}_normal.jpg"
            if os.path.exists(src):
                cv2.imwrite(f"{ASSETS}/{tag}_{who}_scale{sc}.png", cv2.imread(src))

    meta = dict(obj=obj, pair=pair, n_frames=n, n_ref_frames=len(refs),
                epoch=ck.get("epoch"))
    pickle.dump(meta, open(f"{ASSETS}/{tag}_meta.pkl", "wb"))
    print("touch assets written for", tag)


# --------------------------------------------------------------------- match
def do_match(obj, pair, tag, reproj_threshold=8.0):
    """Real SuperPoint + SuperGlue correspondences on the two normal renders."""
    from imcui_match import compute_imcui_sparse_matches

    left = cv2.cvtColor(cv2.imread(f"{REF_RENDER}/{obj}/{pair}_scale25_normal.jpg"),
                        cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    right = cv2.cvtColor(cv2.imread(f"{QUERY_RENDER}/{obj}/{pair}_scale25_normal.jpg"),
                         cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    pts_l, pts_r = compute_imcui_sparse_matches(left, right, "superpoint_superglue")
    # pts are (row, col); RANSAC wants (x, y)
    xy_l = pts_l[:, ::-1].astype(np.float32)
    xy_r = pts_r[:, ::-1].astype(np.float32)
    H, mask = cv2.findHomography(xy_l, xy_r, cv2.RANSAC, reproj_threshold)
    mask = mask.ravel().astype(bool) if mask is not None else np.ones(len(xy_l), bool)
    print(f"{len(xy_l)} matches, {int(mask.sum())} RANSAC inliers")
    pickle.dump(dict(xy_l=xy_l, xy_r=xy_r, inlier=mask, H=H, obj=obj, pair=pair),
                open(f"{ASSETS}/{tag}_matches.pkl", "wb"))
    save_png(f"{ASSETS}/{tag}_match_left.png", left)
    save_png(f"{ASSETS}/{tag}_match_right.png", right)


# ----------------------------------------------------------------- retrieval
def do_retrieval(obj, query_idx, tag):
    """DINOv3 leave-one-out retrieval over one object's 10 touches."""
    from retrieve_touch import load_dino_model, extract_features

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = load_dino_model("dinov3_vitb16", DINO_W, device)

    idxs = sorted(int(f.split("_")[0]) for f in os.listdir(f"{EVAL_RENDER}/{obj}")
                  if f.endswith("_scale25_normal.jpg"))
    paths = [f"{EVAL_RENDER}/{obj}/{i}_scale25_normal.jpg" for i in idxs]
    feats = extract_features(model, transform, paths, device)
    sims = (feats @ feats.T).numpy()

    q = idxs.index(query_idx)
    row = sims[q].copy()
    row[q] = -np.inf                        # leave the query itself out
    order = np.argsort(-row)
    print(f"object {obj}, query touch {query_idx}: top-1 is touch "
          f"{idxs[order[0]]} (cosine similarity {row[order[0]]:.3f})")
    for r in order[:5]:
        print(f"    touch {idxs[r]}: {row[r]:.3f}")

    # thumbnails: geometry render + the deepest-press frame of each touch video
    for i in idxs:
        cv2.imwrite(f"{ASSETS}/{tag}_db{i}_normal.png",
                    cv2.imread(f"{EVAL_RENDER}/{obj}/{i}_scale25_normal.jpg"))
        vid = f"{EVAL_RENDER}/{obj}/{i}_tactile_normal.mp4"
        if os.path.exists(vid):
            frames = read_video(vid)
            if frames:
                save_png(f"{ASSETS}/{tag}_db{i}_touch.png", frames[len(frames) // 2])
    pickle.dump(dict(obj=obj, idxs=idxs, query_idx=query_idx, sims=sims,
                     loo_scores=row, order=[idxs[r] for r in order]),
                open(f"{ASSETS}/{tag}_retrieval.pkl", "wb"))


def do_benchret(obj, query_idx, tag, scale="100"):
    """DINOv3 retrieval for one query of the ground-truth-retrieval benchmark.

    The database is that object's reference touches and the query is the
    held-out query render, so the same touch can illustrate all three method
    steps: whichever reference this picks is the one the rest of the figure
    goes on to align and refine.
    """
    from retrieve_touch import load_dino_model, extract_features

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = load_dino_model("dinov3_vitb16", DINO_W, device)

    # The pipeline retrieves on the 1x renders (transfer_pipeline.py --scale 100,
    # as in job2_full_pipeline/run_transfer.sh); the 4x renders are used later, for
    # feature matching. Keep this identical to the run.
    idxs = sorted(int(f.split("_")[0]) for f in os.listdir(f"{REF_RENDER}/{obj}")
                  if f.endswith(f"_scale{scale}_normal.jpg"))
    ref_feats = extract_features(
        model, transform,
        [f"{REF_RENDER}/{obj}/{i}_scale{scale}_normal.jpg" for i in idxs], device)
    q_feat = extract_features(
        model, transform,
        [f"{QUERY_RENDER}/{obj}/{query_idx}_scale{scale}_normal.jpg"], device)
    scores = (q_feat @ ref_feats.T).numpy()[0]
    order = [idxs[i] for i in np.argsort(-scores)]
    print(f"object {obj}, query {query_idx}: top-1 reference is {order[0]} "
          f"(cosine similarity {scores[idxs.index(order[0])]:.3f}); "
          f"ground-truth pairing is {query_idx}")

    cv2.imwrite(f"{ASSETS}/{tag}_query_normal.png",
                cv2.imread(f"{QUERY_RENDER}/{obj}/{query_idx}_scale{scale}_normal.jpg"))
    for i in idxs:
        cv2.imwrite(f"{ASSETS}/{tag}_db{i}_normal.png",
                    cv2.imread(f"{REF_RENDER}/{obj}/{i}_scale{scale}_normal.jpg"))
        vid = f"{TRANSFER}/{obj}/{i}_ref_tactile_normal.mp4"
        if os.path.exists(vid):
            frames = read_video(vid)
            if frames:
                save_png(f"{ASSETS}/{tag}_db{i}_touch.png", frames[len(frames) // 2])
    pickle.dump(dict(obj=obj, idxs=idxs, query_idx=query_idx, loo_scores=scores,
                     order=order, scale=scale),
                open(f"{ASSETS}/{tag}_retrieval.pkl", "wb"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--do", nargs="+", default=["touch", "match", "retrieval"],
                   choices=["touch", "match", "retrieval", "benchret"])
    p.add_argument("--obj", type=int, default=967)
    p.add_argument("--pair", type=int, default=1)
    p.add_argument("--tag", default=None, help="asset filename prefix; defaults to obj_pair")
    p.add_argument("--ret_obj", type=int, default=3)
    p.add_argument("--ret_query", type=int, default=0)
    p.add_argument("--ret_tag", default=None)
    args = p.parse_args()
    tag = args.tag or f"{args.obj}_{args.pair}"
    ret_tag = args.ret_tag or f"ret{args.ret_obj}_{args.ret_query}"
    os.makedirs(ASSETS, exist_ok=True)

    if "touch" in args.do:
        do_touch(args.obj, args.pair, tag)
    if "match" in args.do:
        do_match(args.obj, args.pair, tag)
    if "retrieval" in args.do:
        do_retrieval(args.ret_obj, args.ret_query, ret_tag)
    if "benchret" in args.do:
        do_benchret(args.obj, args.pair, f"bench{tag}")


if __name__ == "__main__":
    main()
