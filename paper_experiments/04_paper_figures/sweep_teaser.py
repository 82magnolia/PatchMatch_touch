"""Pick teaser examples from the local full-pipeline runs, and export their assets.

Two modes:

  --sheet   score every query touch that run_full_pipeline_local.py produced,
            and stitch a contact sheet of the best candidates so a human can
            look at them before choosing
  --tags    export the per-frame images the teaser needs for the chosen touches,
            under log/paper_job04_paper_figures/assets/fp{obj}_{pair}_*

A good teaser example needs three things beyond a decent score:
  * the touch has to *look* like something (a flat, barely-contacting press
    makes a dull row),
  * the press has to change over time, or the six frames look like six copies,
  * the reference and query geometry have to differ visibly, otherwise the
    analogy looks trivial.
Each is measured directly from the images rather than guessed.
"""
import argparse
import os
import pickle

import cv2
import numpy as np
import torch

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
import sys
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/rebot_net")

SRC = f"{ROOT}/Taxim/results/gen_contact_raw_eval_tactile_normal_pseudo_mini"
FP = f"{ROOT}/log/paper_job04_paper_figures/fullpipe"
ASSETS = f"{ROOT}/log/paper_job04_paper_figures/assets"
SHEET = f"{ROOT}/log/paper_job04_paper_figures/teaser_candidates.png"
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


def save_png(path, rgb01):
    cv2.imwrite(path, cv2.cvtColor((np.clip(rgb01, 0, 1) * 255).round().astype(np.uint8),
                                   cv2.COLOR_RGB2BGR))


def score_looks(obj, pair, ref_idx):
    """Contact strength, how much the press changes, and how unlike the two
    geometry renders are. All three are read off the actual images."""
    gt = read_video(f"{FP}/{obj}/transfer/{pair}_query_tactile_normal.mp4")
    if len(gt) < 10:
        return None
    flat = np.array([0.5, 0.5, 1.0], np.float32)      # the no-contact colour
    dev = [float(np.abs(f - flat).mean()) for f in gt]
    mid = gt[len(gt) // 2]
    rn = cv2.imread(f"{SRC}/{obj}/{ref_idx}_scale25_normal.jpg")
    qn = cv2.imread(f"{SRC}/{obj}/{pair}_scale25_normal.jpg")
    geom_diff = float(np.abs(rn.astype(np.float32) - qn.astype(np.float32)).mean() / 255)
    return dict(contact=max(dev), motion=float(np.std(dev)),
                detail=float(mid.std()), geom_diff=geom_diff, n_frames=len(gt))


def rank(recs, min_psnr):
    out = []
    for r in recs:
        if r["ref_idx"] is None or not np.isfinite(r["psnr_refined"]):
            continue
        if r["psnr_refined"] < min_psnr:
            continue
        s = score_looks(r["obj"], r["pair"], r["ref_idx"])
        if s is None:
            continue
        r = {**r, **s}
        # a plain sum of normalised parts; the human still picks from the sheet
        r["rank_score"] = (min(r["contact"] / 0.10, 1.5) + min(r["motion"] / 0.02, 1.5)
                           + min(r["geom_diff"] / 0.10, 1.0)
                           + min(r["psnr_refined"] / 40.0, 1.0))
        out.append(r)
    out.sort(key=lambda x: -x["rank_score"])
    return out


def contact_sheet(ranked, n, cols=6):
    """One row per candidate: reference geometry, query geometry, then reference,
    coarse, ours and truth at the deepest press."""
    rows = []
    for r in ranked[:n]:
        obj, pair, ref = r["obj"], r["pair"], r["ref_idx"]
        mid = r["n_frames"] // 2
        cells = [cv2.imread(f"{SRC}/{obj}/{ref}_scale25_normal.jpg"),
                 cv2.imread(f"{SRC}/{obj}/{pair}_scale25_normal.jpg")]
        for kind in ("ref_tactile_normal", "transferred", "query_tactile_normal"):
            fr = read_video(f"{FP}/{obj}/transfer/{pair}_{kind}.mp4")
            cells.append(cv2.cvtColor((fr[min(mid, len(fr) - 1)] * 255).astype(np.uint8),
                                      cv2.COLOR_RGB2BGR) if fr else np.zeros((240, 320, 3), np.uint8))
        row = np.hstack([cv2.resize(c, (176, 132)) for c in cells])
        cv2.putText(row, f"obj {obj} q{pair} <- ref {ref}   {r['psnr_refined']:.1f} dB",
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)
    sheet = np.vstack(rows)
    cv2.imwrite(SHEET, sheet)
    print("wrote", SHEET)


def export(obj, pair, ref_idx):
    """Cache every frame the teaser needs for one full-pipeline touch."""
    from dataset import TactileTransferDataset
    from train import build_model

    tag = f"fp{obj}_{pair}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model("rebot_S", cond_chans=3, film_chans=0, bottleneck_hw=24,
                        time_cond="film").to(device)
    ck = torch.load(CKPT, map_location=device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    class Nested(TactileTransferDataset):
        def __init__(self, *a, **k):
            self.NUM_PAIRS = 32
            super().__init__(*a, **k)

        def _obj_dir(self, obj_id):
            return os.path.join(self.transfer_dir, str(obj_id), "transfer")

    ds = Nested(FP, [obj], split="test", cond_dir=SRC, film_modality="normal",
                film_scale=100, geom_concat=True, video_type="tactile_normal",
                time_cond="film")
    preds, gts, coarses = [], [], []
    with torch.no_grad():
        for lq, gt, blank, film, t_norm in ds.iter_video_pairs(obj, pair):
            t_in = torch.tensor([t_norm], device=device)
            pr = model(lq.unsqueeze(0).to(device), film=None, t=t_in).squeeze(0)
            preds.append(pr.cpu().clamp(0, 1).permute(1, 2, 0).numpy())
            gts.append(gt.permute(1, 2, 0).numpy())
            coarses.append(lq[1, :3].permute(1, 2, 0).numpy())
    refs = read_video(f"{FP}/{obj}/transfer/{pair}_ref_tactile_normal.mp4")

    os.makedirs(ASSETS, exist_ok=True)
    for i in range(len(preds)):
        save_png(f"{ASSETS}/{tag}_pred_{i:03d}.png", preds[i])
        save_png(f"{ASSETS}/{tag}_gt_{i:03d}.png", gts[i])
        save_png(f"{ASSETS}/{tag}_coarse_{i:03d}.png", coarses[i])
    for i, fr in enumerate(refs):
        save_png(f"{ASSETS}/{tag}_ref_{i:03d}.png", fr)
    for who, idx in (("refnorm", ref_idx), ("querynorm", pair)):
        for sc in ("100", "25"):
            src = f"{SRC}/{obj}/{idx}_scale{sc}_normal.jpg"
            if os.path.exists(src):
                cv2.imwrite(f"{ASSETS}/{tag}_{who}_scale{sc}.png", cv2.imread(src))

    recs = {(r["obj"], r["pair"]): r for r in pickle.load(open(f"{FP}/candidates.pkl", "rb"))}
    r = recs[(obj, pair)]
    pickle.dump(dict(obj=obj, pair=pair, ref_idx=ref_idx, n_frames=len(preds),
                     n_ref_frames=len(refs), epoch=ck.get("epoch"),
                     psnr_coarse=r["psnr_coarse"], psnr_refined=r["psnr_refined"],
                     benchmark="full pipeline (retrieval included)"),
                open(f"{ASSETS}/{tag}_meta.pkl", "wb"))
    print(f"exported {tag}: reference touch {ref_idx}, {len(preds)} frames, "
          f"{r['psnr_refined']:.1f} dB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--n_sheet", type=int, default=24)
    ap.add_argument("--min_psnr", type=float, default=28.0)
    ap.add_argument("--tags", nargs="*", default=None,
                    help="'obj_pair' touches to export, e.g. 12_3 27_9")
    args = ap.parse_args()

    recs = pickle.load(open(f"{FP}/candidates.pkl", "rb"))
    by = {(r["obj"], r["pair"]): r for r in recs}
    if args.sheet:
        ranked = rank(recs, args.min_psnr)
        print(f"{len(ranked)} of {len(recs)} touches pass {args.min_psnr} dB")
        for r in ranked[:args.n_sheet]:
            print(f"  obj {r['obj']} q{r['pair']} <- ref {r['ref_idx']}: "
                  f"{r['psnr_refined']:.1f} dB, contact {r['contact']:.3f}, "
                  f"motion {r['motion']:.3f}, geometry difference {r['geom_diff']:.3f}")
        contact_sheet(ranked, args.n_sheet)
    for t in args.tags or []:
        o, p = (int(x) for x in t.split("_"))
        export(o, p, by[(o, p)]["ref_idx"])


if __name__ == "__main__":
    main()
