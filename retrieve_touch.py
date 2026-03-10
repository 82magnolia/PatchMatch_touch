"""
Top-K touch retrieval using DINOv2 features.

Given a reference folder and a query folder of touch outputs from gen_contact_video.py,
builds a DINOv2 feature database for the reference entries and retrieves the top-K most
similar reference entries for each query entry based on a chosen static modality.

Usage:
    python retrieve_touch.py \
        --ref_dir Taxim/results/gen_contact \
        --query_dir Taxim/results/gen_contact \
        --modality normal --scale 25 --top_k 5
"""

import argparse
import os
import pickle
import re
from os import path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_files(folder, modality, scale):
    """Return sorted list of (idx, filepath) for all matching entries in folder.

    Matches:
      - `{idx}_scale{scale}_{modality}.jpg`  when scale is given
      - `{idx}_{modality}.jpg`               when scale is None
    """
    if scale is not None:
        pattern = re.compile(rf"^(\d+)_scale{scale}_{re.escape(modality)}\.jpg$")
    else:
        pattern = re.compile(rf"^(\d+)_{re.escape(modality)}\.jpg$")

    entries = []
    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if m:
            idx = int(m.group(1))
            entries.append((idx, osp.join(folder, fname)))

    entries.sort(key=lambda x: x[0])
    return entries


# ---------------------------------------------------------------------------
# DINOv2 model
# ---------------------------------------------------------------------------

def load_dino_model(model_name, device):
    """Load a DINOv2 model from torch hub and return (model, transform)."""
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    return model, transform


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(model, transform, paths, device, batch_size=32):
    """Extract DINOv2 CLS-token features for a list of image paths.

    Returns:
        Tensor of shape (N, D), L2-normalised.
    """
    all_feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="  Extracting features", leave=False):
        batch_paths = paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            img_bgr = cv2.imread(p)
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot read image: {p}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            imgs.append(transform(img_rgb))
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            out = model.forward_features(batch)
            feats = out["x_norm_clstoken"]   # (B, D)
        feats = F.normalize(feats, dim=-1)
        all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0)   # (N, D)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def compute_topk(query_feats, ref_feats, k):
    """Compute top-K reference indices for each query via cosine similarity.

    Args:
        query_feats: (N_q, D) L2-normalised
        ref_feats:   (N_r, D) L2-normalised
        k:           int

    Returns:
        topk_idxs: (N_q, K) int64 tensor — indices into the ref list
        topk_sims: (N_q, K) float tensor — cosine similarities
    """
    # (N_q, N_r)
    sim_matrix = query_feats @ ref_feats.T
    k_eff = min(k, ref_feats.shape[0])
    topk_sims, topk_idxs = sim_matrix.topk(k_eff, dim=-1, largest=True, sorted=True)
    return topk_idxs, topk_sims


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(query_entries, ref_entries, topk_idxs, topk_sims, save_dir):
    """Save retrieval results as a .pkl file.

    The pkl contains a list of dicts, one per query, each with:
      - 'query_idx': int contact-point index in query folder
      - 'topk_ref_indices': list of int contact-point indices in ref folder (rank 1 first)
      - 'topk_similarities': list of float cosine similarity scores
    """
    results = []
    for qi, (q_idx, _) in enumerate(query_entries):
        row = {
            "query_idx": q_idx,
            "topk_ref_indices": [ref_entries[ri][0] for ri in topk_idxs[qi].tolist()],
            "topk_similarities": topk_sims[qi].tolist(),
        }
        results.append(row)

    pkl_path = osp.join(save_dir, "results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved retrieval results to: {pkl_path}")
    return pkl_path


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_figure(query_idx, query_path, ref_entries, topk_ref_list_idxs, topk_sims_list,
                save_dir, modality, k):
    """Save a single-row retrieval figure for one query entry.

    Layout: [Query | Rank-1 | Rank-2 | ... | Rank-K]

    Args:
        query_idx:          int, contact-point index of this query
        query_path:         str, path to query image
        ref_entries:        list of (ref_idx, ref_path) for the full reference set
        topk_ref_list_idxs: list of int — positions into ref_entries (not contact idx)
        topk_sims_list:     list of float
        save_dir:           str
        modality:           str
        k:                  int
    """
    n_cols = 1 + len(topk_ref_list_idxs)
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.5))
    if n_cols == 1:
        axes = [axes]

    def read_rgb(path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Query column
    axes[0].imshow(read_rgb(query_path))
    axes[0].set_title(f"Query #{query_idx}\n({modality})", fontsize=9)
    axes[0].axis("off")

    # Retrieved columns
    for rank, (list_idx, sim) in enumerate(zip(topk_ref_list_idxs, topk_sims_list), start=1):
        ref_idx, ref_path = ref_entries[list_idx]
        axes[rank].imshow(read_rgb(ref_path))
        axes[rank].set_title(f"Ref #{ref_idx}\nrank {rank} | sim={sim:.3f}", fontsize=9)
        axes[rank].axis("off")

    fig.suptitle(f"Query #{query_idx} — Top-{k} retrieval ({modality})", fontsize=10)
    plt.tight_layout()

    out_path = osp.join(save_dir, f"query_{query_idx}_retrieval.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Top-K touch retrieval using DINOv2 features."
    )
    parser.add_argument("--ref_dir", required=True, type=str,
                        help="Path to reference touch outputs folder.")
    parser.add_argument("--query_dir", required=True, type=str,
                        help="Path to query touch outputs folder.")
    parser.add_argument("--modality", required=True,
                        choices=["color", "normal", "curvature", "height"],
                        help="Static modality to use for indexing.")
    parser.add_argument("--scale", default=None, type=int,
                        help="Scale suffix in mm (e.g. 25 for _scale25_). "
                             "Omit to use base-resolution files.")
    parser.add_argument("--top_k", default=5, type=int,
                        help="Number of top retrievals per query entry.")
    parser.add_argument("--save_dir", default="./log/touch_retrieval", type=str,
                        help="Output directory for results and figures.")
    parser.add_argument("--dino_model", default="dinov2_vits14",
                        choices=["dinov2_vits14", "dinov2_vitb14",
                                 "dinov2_vitl14", "dinov2_vitg14"],
                        help="DINOv2 model variant.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Discover files
    # -----------------------------------------------------------------------
    print(f"Discovering reference files in: {args.ref_dir}")
    ref_entries = discover_files(args.ref_dir, args.modality, args.scale)
    if not ref_entries:
        scale_str = f"_scale{args.scale}_" if args.scale else "_"
        raise FileNotFoundError(
            f"No files matching '{{}}{scale_str}{args.modality}.jpg' found in {args.ref_dir}"
        )
    print(f"  Found {len(ref_entries)} reference entries.")

    print(f"Discovering query files in: {args.query_dir}")
    query_entries = discover_files(args.query_dir, args.modality, args.scale)
    if not query_entries:
        scale_str = f"_scale{args.scale}_" if args.scale else "_"
        raise FileNotFoundError(
            f"No files matching '{{}}{scale_str}{args.modality}.jpg' found in {args.query_dir}"
        )
    print(f"  Found {len(query_entries)} query entries.")

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading DINOv2 model '{args.dino_model}' on {device}...")
    model, transform = load_dino_model(args.dino_model, device)

    # -----------------------------------------------------------------------
    # Extract features
    # -----------------------------------------------------------------------
    ref_paths = [p for _, p in ref_entries]
    query_paths = [p for _, p in query_entries]

    print("Extracting reference features...")
    ref_feats = extract_features(model, transform, ref_paths, device)   # (N_r, D)

    print("Extracting query features...")
    query_feats = extract_features(model, transform, query_paths, device)  # (N_q, D)

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------
    print(f"Computing top-{args.top_k} retrievals...")
    topk_idxs, topk_sims = compute_topk(query_feats, ref_feats, args.top_k)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    save_results(query_entries, ref_entries, topk_idxs, topk_sims, args.save_dir)

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    print("Generating retrieval figures...")
    for qi, (q_idx, q_path) in enumerate(tqdm(query_entries, desc="Figures")):
        list_idxs = topk_idxs[qi].tolist()
        sims = topk_sims[qi].tolist()
        make_figure(
            query_idx=q_idx,
            query_path=q_path,
            ref_entries=ref_entries,
            topk_ref_list_idxs=list_idxs,
            topk_sims_list=sims,
            save_dir=args.save_dir,
            modality=args.modality,
            k=args.top_k,
        )

    print(f"Done. Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
