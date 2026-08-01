"""Run the pretrained refinement network over the ground-truth-retrieval eval objects.

For every touch location of objects 951-1000 this script

  * loads the coarse-transferred video (produced by
    main_retrieval_transfer_feat_match.py),
  * runs the pretrained ReBotNet-S refinement network
    (log/rebot_checkpoints_S_geomcat_film/best.pth -- normal-map concatenation +
    sinusoidal temporal FiLM),
  * scores both the coarse and the refined video against the ground-truth query
    video (MSE / PSNR / SSIM / LPIPS, averaged over frames),
  * and caches the middle frame of every video stream as a PNG so the
    qualitative figure can be assembled without re-running the network.

Unlike rebot_net/eval.py this does not assume the transfer directory contains
all 1000 objects (it takes an explicit object-id range), which is what lets it
run on a partial, locally-computed transfer directory.
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np
import torch

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
sys.path.insert(0, f"{ROOT}/rebot_net")
from dataset import TactileTransferDataset          # noqa: E402
from train import build_model                       # noqa: E402

TRANSFER = f"{ROOT}/log/transfer_feat_match_pseudo_mini_tactile_normal_superpoint_superglue"
COND = f"{ROOT}/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini"
REF_RENDER = f"{ROOT}/Taxim/results/gen_contact_full_tactile_normal_pseudo_mini"
CKPT = f"{ROOT}/log/rebot_checkpoints_S_geomcat_film/best.pth"
RETRIEVAL = f"{ROOT}/log/touch_retrieval"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--first_obj", type=int, default=951)
    p.add_argument("--last_obj", type=int, default=1000)
    p.add_argument("--out_dir", default=f"{ROOT}/log/paper_job02_gt_retrieval_figure")
    p.add_argument("--ckpt", default=CKPT)
    return p.parse_args()


def mid_index(n):
    """Middle frame of the press cycle -- the deepest press of back_forth_press."""
    return n // 2


def to_u8(x):
    return (np.clip(x, 0, 1) * 255.0).round().astype(np.uint8)


def save_png(path, rgb01):
    cv2.imwrite(path, cv2.cvtColor(to_u8(rgb01), cv2.COLOR_RGB2BGR))


def read_frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    asset_dir = os.path.join(args.out_dir, "assets")
    os.makedirs(asset_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # geomcat_film: the query normal render enters as 3 concatenated input
    # channels (cond_chans=3, no FiLM channels) and time enters via FiLM.
    model = build_model("rebot_S", cond_chans=3, film_chans=0,
                        bottleneck_hw=24, time_cond="film").to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    print(f"loaded {args.ckpt} @ epoch {ck.get('epoch')} (val PSNR {ck.get('best_psnr', 0):.2f})")

    import lpips
    from skimage.metrics import mean_squared_error as c_mse
    from skimage.metrics import peak_signal_noise_ratio as c_psnr
    from skimage.metrics import structural_similarity as c_ssim
    lpips_model = lpips.LPIPS(net="alex").to(device)
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    obj_ids = [o for o in range(args.first_obj, args.last_obj + 1)
               if os.path.isdir(os.path.join(TRANSFER, str(o)))]
    print(f"{len(obj_ids)} objects available in the transfer directory")

    ds = TactileTransferDataset(TRANSFER, obj_ids, split="test", cond_dir=COND,
                                film_modality="normal", film_scale=100,
                                geom_concat=True, video_type="tactile_normal",
                                time_cond="film")

    def lpips_of(a, b):
        ta = torch.from_numpy(a).permute(2, 0, 1)[None].to(device) * 2 - 1
        tb = torch.from_numpy(b).permute(2, 0, 1)[None].to(device) * 2 - 1
        return float(lpips_model(ta, tb).item())

    records = []
    for obj in obj_ids:
        # which reference touch the retrieval step picked for each query
        ref_of_query = {}
        rp = os.path.join(RETRIEVAL, str(obj), "results.pkl")
        if os.path.exists(rp):
            for e in pickle.load(open(rp, "rb")):
                ref_of_query[int(e["query_idx"])] = int(e["topk_ref_indices"][0])

        for pair in range(ds.NUM_PAIRS):
            if not ds.lq_video_exists(obj, pair):
                continue
            preds, gts, coarses = [], [], []
            with torch.no_grad():
                for lq, gt, blank, film, t_norm in ds.iter_video_pairs(obj, pair):
                    t_in = torch.tensor([t_norm], device=device)
                    pr = model(lq.unsqueeze(0).to(device), film=None, t=t_in).squeeze(0)
                    preds.append(pr.cpu().clamp(0, 1).permute(1, 2, 0).numpy())
                    gts.append(gt.permute(1, 2, 0).numpy())
                    # lq channels 0:3 of the *current* frame are the coarse transfer
                    coarses.append(lq[1, :3].permute(1, 2, 0).numpy())
            if not preds:
                continue

            def score(seq):
                m = [c_mse(g, s) for g, s in zip(gts, seq)]
                # A no-contact frame can be reproduced exactly (both sides are the
                # constant flat-normal colour), giving MSE 0 and infinite PSNR.
                # rebot_net/eval.py caps those frames at 100 dB; match that.
                p = [c_psnr(g, s, data_range=1.0) if mse > 0 else 100.0
                     for g, s, mse in zip(gts, seq, m)]
                s_ = [c_ssim(g, s, channel_axis=2, data_range=1.0) for g, s in zip(gts, seq)]
                l = [lpips_of(g, s) for g, s in zip(gts, seq)]
                return dict(MSE=float(np.mean(m)), PSNR=float(np.mean(p)),
                            SSIM=float(np.mean(s_)), LPIPS=float(np.mean(l)))

            sc_coarse, sc_ref = score(coarses), score(preds)
            k = mid_index(len(preds))

            base = os.path.join(asset_dir, f"{obj}_{pair}")
            save_png(f"{base}_04_coarse.png", coarses[k])
            save_png(f"{base}_05_refined.png", preds[k])
            save_png(f"{base}_06_gt_query.png", gts[k])
            rf = read_frame(os.path.join(TRANSFER, str(obj), f"{pair}_ref_tactile_normal.mp4"), k)
            if rf is not None:
                save_png(f"{base}_01_ref_touch.png", rf)
            ridx = ref_of_query.get(pair, pair)
            for tag, src in (("02_ref_normal", f"{REF_RENDER}/{obj}/{ridx}_scale100_normal.jpg"),
                             ("03_query_normal", f"{COND}/{obj}/{pair}_scale100_normal.jpg")):
                if os.path.exists(src):
                    im = cv2.imread(src)
                    cv2.imwrite(f"{base}_{tag}.png", im)

            records.append(dict(obj=obj, pair=pair, ref_idx=ridx, n_frames=len(preds),
                                mid_frame=k, coarse=sc_coarse, refined=sc_ref))
            print(f"{obj}_{pair}: coarse PSNR {sc_coarse['PSNR']:.2f} -> "
                  f"refined {sc_ref['PSNR']:.2f}  (ref touch {ridx})", flush=True)

    with open(os.path.join(args.out_dir, "per_touch_metrics.pkl"), "wb") as f:
        pickle.dump(records, f)

    def agg(key):
        return {m: float(np.mean([r[key][m] for r in records]))
                for m in ("MSE", "PSNR", "SSIM", "LPIPS")}

    summary = dict(n_touches=len(records), n_objects=len(obj_ids),
                   checkpoint=args.ckpt, epoch=ck.get("epoch"),
                   coarse=agg("coarse"), refined=agg("refined"))
    with open(os.path.join(args.out_dir, "summary.pkl"), "wb") as f:
        pickle.dump(summary, f)
    print("\n=== SUMMARY ===")
    print(summary)


if __name__ == "__main__":
    main()
