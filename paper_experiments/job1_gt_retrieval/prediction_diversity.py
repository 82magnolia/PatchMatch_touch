"""How much does each method's prediction actually change with the query?

For one touch index, take the middle frame of the prediction for every object and
measure the mean absolute pixel difference between predictions for DIFFERENT
objects. A method that ignores its conditioning returns the same picture every
time, so this collapses towards 0. Each method is compared against the ground
truth computed on THAT METHOD'S OWN set of objects, because our refined videos
are only saved for a subset.
"""
import cv2, itertools, json, numpy as np, os
ROOT="/data1/junhokim/Projects/PatchMatch_touch"
OBJS=list(range(951,1001)); TOUCH=0
GT="log/paper_job1_transfer_normal/{o}/%d_query_tactile_normal.mp4"%TOUCH
SRC={
 "TaRF (epoch 5, finetuned)": "log/paper_job1_baselines/tarf/{o}/transfer/%d_transferred.mp4"%TOUCH,
 "TaRF (epoch 29, from scratch)": "log/paper_job1_baselines/tarf_v2/{o}/transfer/%d_transferred.mp4"%TOUCH,
 "TaRF (epoch 29, finetuned)": "log/paper_job1_baselines/tarf_v3/{o}/transfer/%d_transferred.mp4"%TOUCH,
 "Tactile Normal Quilting": "log/paper_job1_baselines/quilting/{o}/transfer/%d_transferred.mp4"%TOUCH,
 "ObjectFolder INR": "log/paper_job1_baselines/inr/{o}/transfer/%d_transferred.mp4"%TOUCH,
 "Ours (coarse transfer, normals)": "log/paper_job1_transfer_normal/{o}/%d_transferred.mp4"%TOUCH,
 "Ours (refined, normals)": "log/paper_job1_refine_ours_normal/videos/{o}_%d_enhanced.mp4"%TOUCH,
}
def mid(p):
    if not os.path.exists(p): return None
    c=cv2.VideoCapture(p); n=int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    if n<=0: c.release(); return None
    c.set(cv2.CAP_PROP_POS_FRAMES,n//2); ok,f=c.read(); c.release()
    return cv2.resize(f,(160,120)).astype(np.float32)/255.0 if ok else None

def spread(images):
    pairs=list(itertools.combinations(range(len(images)),2))
    rng=np.random.RandomState(0)
    sel=[pairs[i] for i in rng.choice(len(pairs),min(400,len(pairs)),replace=False)]
    return float(np.mean([np.abs(images[a]-images[b]).mean() for a,b in sel]))

res={}
print(f"{'Method':34s} {'n':>3s} {'spread':>7s} {'GT same objs':>13s} {'% of GT':>8s}")
print("-"*70)
for name,t in SRC.items():
    imgs, gts = [], []
    for o in OBJS:
        im=mid(os.path.join(ROOT,t.format(o=o)))
        g=mid(os.path.join(ROOT,GT.format(o=o)))
        if im is not None and g is not None:
            imgs.append(im); gts.append(g)
    if len(imgs)<5: continue
    s, sg = spread(imgs), spread(gts)
    res[name]={"n_objects":len(imgs),"spread":s,"ground_truth_spread_same_objects":sg,
               "percent_of_ground_truth":100*s/sg}
    print(f"{name:34s} {len(imgs):3d} {s:7.4f} {sg:13.4f} {100*s/sg:7.1f}%")
out=os.path.join(ROOT,"paper_experiments/job1_gt_retrieval/prediction_diversity.json")
json.dump({"touch_index":TOUCH,"note":"mean abs difference between middle frames of "
           "different objects' predictions; compared to ground truth on the same objects",
           "methods":res}, open(out,"w"), indent=2)
print(f"\n-> {out}")
