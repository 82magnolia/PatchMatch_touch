"""Folder of symlinks behind the two reconstruction PDFs."""
import os, pickle, sys
ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
JOB = f"{ROOT}/log/paper_job05_recon_figure"
DEST = f"{JOB}/figure_assets/recon_pdf"
ROWS = {"row1": "prediction", "row2": "relief_3d", "row3": "simulated_rgb"}
TRANSFER = f"{ROOT}/log/transfer_feat_match_pseudo_mini_tactile_normal_superpoint_superglue_normalmatch"
CKPT = f"{ROOT}/log/rebot_checkpoints_S_geomcat_film/best.pth"
QR = f"{ROOT}/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini"

def link(src, dst, man):
    if not os.path.exists(src):
        man.append(f"  MISSING {os.path.relpath(dst, DEST)} <- {src}"); return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst): os.remove(dst)
    os.symlink(src, dst); man.append(f"  {os.path.relpath(dst, DEST)}\n      -> {src}")

recs = pickle.load(open(f"{ROOT}/log/paper_job02_gt_retrieval_figure_normalmatch/per_touch_metrics.pkl","rb"))
score = {(r["obj"], r["pair"]): r for r in recs}
head_all = []
for obj, pair in [(993, 2), (994, 3)]:
    tag = f"{obj}_{pair}"; out = f"{DEST}/{tag}"; man = []
    link(f"{JOB}/recon_{tag}.pdf", f"{out}/figure/recon_{tag}.pdf", man)
    link(f"{JOB}/recon_{tag}.png", f"{out}/figure/recon_{tag}.png", man)
    man.append("\ncells/")
    for name in sorted(os.listdir(f"{JOB}/assets_pdf/{tag}")):
        row = next((k for k in ROWS if f"_{k}_" in name), None)
        if row is None: continue
        col, frame = name.split("_")[0], name.split("_")[1]
        link(f"{JOB}/assets_pdf/{tag}/{name}", f"{out}/cells/{ROWS[row]}/{col}_{frame}.png", man)
    man.append("\nobject_renders/")
    for sub in ("", "closeup", "marked"):
        d = f"{JOB}/object_renders/{sub}".rstrip("/")
        if not os.path.isdir(d): continue
        for name in sorted(os.listdir(d)):
            if name.startswith(str(obj)) and name.endswith(".png"):
                link(f"{d}/{name}", f"{out}/object_renders/{sub or 'views'}/{name}", man)
    man.append("\nsources/")
    link(f"{TRANSFER}/{obj}/{pair}_transferred.mp4", f"{out}/sources/coarse_transfer_video.mp4", man)
    link(f"{TRANSFER}/{obj}/{pair}_ref_tactile_normal.mp4", f"{out}/sources/reference_touch_video.mp4", man)
    link(f"{TRANSFER}/{obj}/{pair}_query_tactile_normal.mp4", f"{out}/sources/ground_truth_query_video.mp4", man)
    for sc in ("100","25"):
        link(f"{QR}/{obj}/{pair}_scale{sc}_normal.jpg", f"{out}/sources/query_normal_scale{sc}.jpg", man)
    link(CKPT, f"{out}/sources/refinement_checkpoint.pth", man)
    r = score[(obj, pair)]
    head = [f"3D reconstruction figure - object {obj}, touch {pair}", "",
            f"  benchmark          ground-truth retrieval (objects 951-1000); the reference is",
            f"                     the touch the benchmark pairs with this query",
            f"  coarse transfer    {r['coarse']['PSNR']:.1f} dB",
            f"  refined (row 1)    {r['refined']['PSNR']:.1f} dB",
            f"  rows               1 prediction, 2 3D relief, 3 simulated colour image",
            f"                     rows 2 and 3 are computed from row 1 alone, no ground truth",
            f"  figure             recon_{tag}.pdf, 7.0 in wide, white background, black text",
            "", "Everything below is a symlink.", ""]
    open(f"{out}/MANIFEST.txt","w").write("\n".join(head+man)+"\n")
    head_all.append("\n".join(head[:8]))
print("\n\n".join(head_all)); print("->", DEST)
