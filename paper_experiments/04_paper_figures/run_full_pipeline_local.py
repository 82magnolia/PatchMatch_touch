"""Run the complete pipeline locally on part of the full-pipeline benchmark.

The teaser should be an example of the *whole* system working, retrieval
included, not of the ground-truth-retrieval benchmark where the reference is
handed to the method. That benchmark's runs live on the other machine, so this
script reproduces them here for a handful of objects:

  1. materialise each object's reference / query split as symlink directories,
     using the very same split the paper reports
     (paper_experiments/job2_full_pipeline/splits.json, seed 0),
  2. run transfer_pipeline.py -- DINOv3 retrieval over the reference touches,
     then SuperPoint + SuperGlue coarse alignment on surface normals at 4x the
     sensor footprint (identical flags to job2_full_pipeline/run_transfer.sh),
  3. refine every query touch with the paper's network and score it against the
     held-out ground truth.

Output: log/paper_job04_paper_figures/fullpipe/{obj}/transfer/… plus
candidates.pkl, one record per query touch, which sweep_teaser.py then ranks.
"""
import argparse
import json
import os
import pickle
import re
import subprocess
import sys

import cv2
import numpy as np
import torch

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/rebot_net")

SRC = f"{ROOT}/Taxim/results/gen_contact_raw_eval_tactile_normal_pseudo_mini"
SPLITS = f"{ROOT}/paper_experiments/job2_full_pipeline/splits.json"
OUT = f"{ROOT}/log/paper_job04_paper_figures/fullpipe"
BENCH = f"{OUT}/_bench"

# The other benchmark: objects 951-1000 keep their reference and query touches in
# two separate directories already, so no split has to be built. Running the same
# pipeline on them means retrieval has to choose among that object's own eight
# reference touches instead of being handed the paired one.
GT_REF = f"{ROOT}/Taxim/results/gen_contact_full_tactile_normal_pseudo_mini"
GT_QUERY = f"{ROOT}/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini"
GT_OUT = f"{ROOT}/log/paper_job04_paper_figures/fullpipe_gtbench"
CKPT = f"{ROOT}/log/rebot_checkpoints_S_geomcat_film/best.pth"
DINO = f"{ROOT}/dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"


def link_touch(src_dir, dst_dir, idx):
    """Symlink every per-touch file of `idx`, keeping the original name.

    The original index has to survive: the refinement network looks its query
    normal render up as {cond_dir}/{obj}/{idx}_scale100_normal.jpg.
    """
    pat = re.compile(rf"^{idx}(?=[_.])")
    for name in os.listdir(src_dir):
        if pat.match(name):
            dst = os.path.join(dst_dir, name)
            if not os.path.exists(dst):
                os.symlink(os.path.join(src_dir, name), dst)


def build_split(obj, split):
    obj_dir = f"{SRC}/{obj}"
    for kind in ("ref", "query"):
        d = f"{BENCH}/{obj}/{kind}"
        os.makedirs(d, exist_ok=True)
        for idx in split[kind]:
            link_touch(obj_dir, d, idx)
    return f"{BENCH}/{obj}"


def run_transfer(obj, ref_dir, query_dir, out_root, pin_ref=None):
    """Run retrieval + coarse alignment for one object.

    pin_ref fixes which reference touch every query is transferred from, instead
    of letting DINOv3 choose. That is for controlled figures -- holding the
    reference constant while the query moves -- and must be labelled as such
    wherever the output is used.
    """
    save = f"{out_root}/{obj}"
    if os.path.exists(f"{save}/transfer/metrics.pkl"):
        print(f"  object {obj}: coarse transfer already done")
        return save
    os.makedirs(save, exist_ok=True)
    retrieval_flags = ["--retrieval_mode", "dinov3", "--retrieval_modality", "normal",
                       "--dino_weights", DINO]
    if pin_ref is not None:
        queries = sorted(int(f.split("_")[0]) for f in os.listdir(query_dir)
                         if f.endswith("_tactile_normal.mp4"))
        tsv = f"{save}/pinned_ref{pin_ref}.tsv"
        with open(tsv, "w") as f:
            f.write("query\tref\n")
            for q in queries:
                f.write(f"{q}\t{pin_ref}\n")
        retrieval_flags = ["--retrieval_mode", "tsv", "--tsv", tsv]
    cmd = [sys.executable, f"{ROOT}/transfer_pipeline.py",
           "--ref_dir", ref_dir, "--query_dir", query_dir,
           "--save_dir", save,
           "--scale", "100",
           "--match_scale", "25", "--match_scale_convention", "obj_scale_factor",
           *retrieval_flags,
           "--transfer_backend", "dinov3_feat_match",
           "--transfer_modality", "normal",
           "--transfer_matcher", "superpoint_superglue",
           "--transfer_offset_matcher", "superpoint_superglue",
           "--transfer_offset_method", "median",
           "--video_type", "tactile_normal",
           "--skip_refine", "--skip_viz"]
    with open(f"{save}/pipeline.log", "w") as log:
        r = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"  object {obj}: FAILED, see {save}/pipeline.log")
        return None
    return save


def make_dataset(objs, out_root=OUT, cond_root=SRC):
    """rebot_net's dataset over transfer_pipeline.py's nested output layout."""
    from dataset import TactileTransferDataset

    class Nested(TactileTransferDataset):
        def __init__(self, *a, **k):
            self.NUM_PAIRS = 32       # benchmark objects have up to 31 touches
            super().__init__(*a, **k)

        def _obj_dir(self, obj_id):
            return os.path.join(self.transfer_dir, str(obj_id), "transfer")

    return Nested(out_root, objs, split="test", cond_dir=cond_root, film_modality="normal",
                  film_scale=100, geom_concat=True, video_type="tactile_normal",
                  time_cond="film")


def psnr_seq(gts, preds):
    from skimage.metrics import peak_signal_noise_ratio as c_psnr
    vals = []
    for g, s in zip(gts, preds):
        mse = float(np.mean((g - s) ** 2))
        if mse <= 1e-12:              # a no-contact frame can match exactly
            continue
        vals.append(c_psnr(g, s, data_range=1.0))
    return float(np.mean(vals)) if vals else float("nan")


def refine_objects(objs, queries_of, out_root=OUT, cond_root=SRC):
    from train import build_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model("rebot_S", cond_chans=3, film_chans=0, bottleneck_hw=24,
                        time_cond="film").to(device)
    ck = torch.load(CKPT, map_location=device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    ds = make_dataset(objs, out_root, cond_root)
    records = []
    for obj in objs:
        retr = {}
        rp = f"{out_root}/{obj}/retrieval/results.pkl"
        if os.path.exists(rp):
            for e in pickle.load(open(rp, "rb")):
                retr[int(e["query_idx"])] = int(e["topk_ref_indices"][0])
        for pair in queries_of(obj):
            if not ds.lq_video_exists(obj, pair):
                continue
            preds, gts, coarses = [], [], []
            with torch.no_grad():
                for lq, gt, blank, film, t_norm in ds.iter_video_pairs(obj, pair):
                    t_in = torch.tensor([t_norm], device=device)
                    pr = model(lq.unsqueeze(0).to(device), film=None, t=t_in).squeeze(0)
                    preds.append(pr.cpu().clamp(0, 1).permute(1, 2, 0).numpy())
                    gts.append(gt.permute(1, 2, 0).numpy())
                    coarses.append(lq[1, :3].permute(1, 2, 0).numpy())
            if not preds:
                continue
            records.append(dict(obj=obj, pair=pair, ref_idx=retr.get(pair),
                                n_frames=len(preds),
                                psnr_coarse=psnr_seq(gts, coarses),
                                psnr_refined=psnr_seq(gts, preds)))
            print(f"  object {obj} touch {pair}: reference {retr.get(pair)}, "
                  f"coarse {records[-1]['psnr_coarse']:.1f} dB -> refined "
                  f"{records[-1]['psnr_refined']:.1f} dB")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", type=int, nargs="*", default=None,
                    help="benchmark object ids; default: the first --n_objects")
    ap.add_argument("--n_objects", type=int, default=25)
    ap.add_argument("--skip_transfer", action="store_true")
    ap.add_argument("--query_root", default=None,
                    help="with --source gt_bench: use these query touches instead of the "
                         "benchmark's own (e.g. the re-simulated, shifted ones written by "
                         "shift_query_touch.py). Must hold {obj}/ subdirectories.")
    ap.add_argument("--out_root", default=None, help="override the output directory")
    ap.add_argument("--pin_ref", type=int, default=None,
                    help="force every query to be transferred from this reference touch "
                         "instead of retrieving one (for controlled figures)")
    ap.add_argument("--source", default="raw_eval", choices=["raw_eval", "gt_bench"],
                    help="'raw_eval' is the full-pipeline benchmark (objects 1-100); "
                         "'gt_bench' runs the same pipeline on the ground-truth-retrieval "
                         "benchmark objects (951-1000), letting retrieval choose the reference")
    args = ap.parse_args()

    if args.source == "gt_bench":
        objs = args.objects or list(range(951, 951 + args.n_objects))
        query_root = args.query_root or GT_QUERY
        out_root = args.out_root or (GT_OUT if args.query_root is None
                                     else f"{GT_OUT}_shifted")
        cond_root = query_root
        os.makedirs(out_root, exist_ok=True)
        done = []
        for obj in objs:
            n_ref = len([f for f in os.listdir(f"{GT_REF}/{obj}")
                         if f.endswith("_scale25_normal.jpg")])
            print(f"object {obj}: {n_ref} references, retrieval chooses among them")
            if args.skip_transfer or run_transfer(obj, f"{GT_REF}/{obj}",
                                                  f"{query_root}/{obj}", out_root,
                                                  args.pin_ref):
                done.append(obj)

        def queries_of(obj):
            return sorted(int(f.split("_")[0]) for f in os.listdir(f"{query_root}/{obj}")
                          if f.endswith("_tactile_normal.mp4"))
    else:
        splits = json.load(open(SPLITS))["objects"]
        objs = args.objects or sorted((int(o) for o in splits), key=int)[:args.n_objects]
        out_root, cond_root = OUT, SRC
        os.makedirs(out_root, exist_ok=True)
        done = []
        for obj in objs:
            s = splits[str(obj)]
            print(f"object {obj}: {len(s['ref'])} references, {len(s['query'])} queries")
            bench_dir = build_split(obj, s)
            if args.skip_transfer or run_transfer(obj, f"{bench_dir}/ref",
                                                  f"{bench_dir}/query", out_root):
                done.append(obj)

        def queries_of(obj):
            return splits[str(obj)]["query"]

    recs = refine_objects(done, queries_of, out_root, cond_root)
    pickle.dump(recs, open(f"{out_root}/candidates.pkl", "wb"))
    print(f"\n{len(recs)} query touches scored -> {out_root}/candidates.pkl")


if __name__ == "__main__":
    main()
