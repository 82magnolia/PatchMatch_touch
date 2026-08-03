"""Hold the query fixed, try each shifted reference touch, and score the result.

The reconstruction figure transfers one reference touch to one query location.
The query is the interesting one (object 993, touch 2, whose fine slats are the
point of the figure), so it stays exactly as the benchmark has it; what varies is
where the *reference* touch was taken. shift_query_touch.py --which ref has
already re-simulated a fan of reference touches around the original spot; this
runs the transfer once per candidate, refines the query, and scores it, so a
good one can be chosen on evidence rather than by guessing.

Each candidate gets its own coarse-alignment run with retrieval pinned to it
(there is only one reference in play, so nothing is left for retrieval to
decide).

    python sweep_reference_shift.py --obj 993 --query 2 \\
        --ref_dir <shifted>/993 --candidates 8 9 10 ... --sheet
"""
import argparse
import os
import pickle
import subprocess
import sys

import cv2
import numpy as np

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/rebot_net")
sys.path.insert(0, f"{ROOT}/paper_experiments/04_paper_figures")
from run_full_pipeline_local import make_dataset, psnr_seq   # noqa: E402

QUERY_RENDER = f"{ROOT}/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini"
OUT = f"{ROOT}/log/paper_job05_recon_figure/refsweep"
CKPT = f"{ROOT}/log/rebot_checkpoints_S_geomcat_film/best.pth"


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


def run_transfer(obj, query, ref_dir, cand):
    """Coarse alignment of one reference candidate onto the fixed query."""
    save = f"{OUT}/cand{cand:02d}/{obj}"
    if os.path.exists(f"{save}/transfer/metrics.pkl"):
        return save
    os.makedirs(save, exist_ok=True)
    tsv = f"{OUT}/cand{cand:02d}/pin.tsv"
    with open(tsv, "w") as f:
        f.write(f"query\tref\n{query}\t{cand}\n")
    cmd = [sys.executable, f"{ROOT}/transfer_pipeline.py",
           "--ref_dir", ref_dir, "--query_dir", f"{QUERY_RENDER}/{obj}",
           "--save_dir", save,
           "--scale", "100",
           "--match_scale", "25", "--match_scale_convention", "obj_scale_factor",
           "--retrieval_mode", "tsv", "--tsv", tsv,
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
        print(f"  candidate {cand}: transfer FAILED, see {save}/pipeline.log")
        return None
    return save


def refine_one(model, obj, query, out_root):
    ds = make_dataset([obj], out_root, QUERY_RENDER)
    if not ds.lq_video_exists(obj, query):
        return None
    import torch
    preds, gts, coarses = [], [], []
    with torch.no_grad():
        for lq, gt, blank, film, t_norm in ds.iter_video_pairs(obj, query):
            t_in = torch.tensor([t_norm], device=next(model.parameters()).device)
            pr = model(lq.unsqueeze(0).to(t_in.device), film=None, t=t_in).squeeze(0)
            preds.append(pr.cpu().clamp(0, 1).permute(1, 2, 0).numpy())
            gts.append(gt.permute(1, 2, 0).numpy())
            coarses.append(lq[1, :3].permute(1, 2, 0).numpy())
    if not preds:
        return None
    return dict(psnr_coarse=psnr_seq(gts, coarses), psnr_refined=psnr_seq(gts, preds),
                n_frames=len(preds))


def sheet(records, obj, query, ref_dir, path):
    """One row per candidate: shifted reference geometry, its touch, the coarse
    transfer, our prediction, and the fixed ground truth for comparison."""
    rows = []
    for r in records:
        c = r["cand"]
        mid = r["n_frames"] // 2
        tdir = f"{OUT}/cand{c:02d}/{obj}/transfer"
        cells = [cv2.imread(f"{ref_dir}/{c}_scale25_normal.jpg")]
        for kind in ("ref_tactile_normal", "transferred", "query_tactile_normal"):
            fr = read_video(f"{tdir}/{query}_{kind}.mp4")
            cells.append(cv2.cvtColor((fr[min(mid, len(fr) - 1)] * 255).astype(np.uint8),
                                      cv2.COLOR_RGB2BGR)
                         if fr else np.zeros((240, 320, 3), np.uint8))
        row = np.hstack([cv2.resize(x, (200, 150)) for x in cells])
        cv2.putText(row, f"cand {c}: {r['moved_mm']:.1f} mm, dir {r['direction_deg']} deg"
                         f"   coarse {r['psnr_coarse']:.1f} -> refined {r['psnr_refined']:.1f} dB",
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)
    cv2.imwrite(path, np.vstack(rows))
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", type=int, default=993)
    ap.add_argument("--query", type=int, default=2)
    ap.add_argument("--shift_dir", required=True,
                    help="directory written by shift_query_touch.py --which ref")
    ap.add_argument("--candidates", type=int, nargs="*", default=None,
                    help="reference indices to try; default: every shifted one")
    ap.add_argument("--sheet", action="store_true")
    args = ap.parse_args()

    ref_dir = f"{args.shift_dir}/{args.obj}"
    labels = {int(m["index"]): m for m in
              np.load(f"{args.shift_dir}/labels.npy", allow_pickle=True)}
    anchors_ply = f"{args.shift_dir}/contact_points.ply"
    import open3d as o3d
    pts = np.asarray(o3d.io.read_point_cloud(anchors_ply).points)
    a = int((pts[:min(labels)].max(axis=0) - pts[:min(labels)].min(axis=0)).argmax())
    ext = float(pts[:min(labels), a].max() - pts[:min(labels), a].min())
    cands = args.candidates or sorted(labels)

    from train import build_model
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model("rebot_S", cond_chans=3, film_chans=0, bottleneck_hw=24,
                        time_cond="film").to(device)
    ck = torch.load(CKPT, map_location=device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    os.makedirs(OUT, exist_ok=True)
    records = []
    for c in cands:
        save = run_transfer(args.obj, args.query, ref_dir, c)
        if save is None:
            continue
        r = refine_one(model, args.obj, args.query, f"{OUT}/cand{c:02d}")
        if r is None:
            print(f"  candidate {c}: no transferred video (matching probably failed)")
            continue
        m = labels[c]
        r.update(cand=c, moved_mm=m["moved_mm_of_bbox"] / ext * 100.0,
                 direction_deg=m["direction_deg"], dist_frac=m["dist_frac"])
        records.append(r)
        print(f"  candidate {c}: {r['moved_mm']:.1f} mm, direction {r['direction_deg']} deg"
              f"  ->  coarse {r['psnr_coarse']:.1f} dB, refined {r['psnr_refined']:.1f} dB")

    pickle.dump(records, open(f"{OUT}/candidates.pkl", "wb"))
    records.sort(key=lambda r: -r["psnr_refined"])
    print("\nbest first:")
    for r in records:
        print(f"  cand {r['cand']:2d}  {r['moved_mm']:4.1f} mm  dir {r['direction_deg']:3d}"
              f"  refined {r['psnr_refined']:.1f} dB")
    if args.sheet:
        sheet(sorted(records, key=lambda r: r["cand"]), args.obj, args.query, ref_dir,
              f"{OUT}/candidates.png")


if __name__ == "__main__":
    main()
