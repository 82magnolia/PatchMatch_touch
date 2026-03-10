"""
Top-K touch retrieval — DINOv2 or pre-specified TSV.

Two retrieval modes selected via --retrieval_mode:

  dinov2 (default):
    Builds a DINOv2 feature database for the reference entries and retrieves
    the top-K most similar ones for each query.

    python retrieve_touch.py \
        --ref_dir Taxim/results/gen_contact \
        --query_dir Taxim/results/gen_contact \
        --modality normal --scale 25 --top_k 5 \
        --retrieval_mode dinov2

  tsv:
    Loads pre-specified retrieval results from a TSV file (--tsv). The TSV
    must have a header "query\\tref" and one row per query with tab-separated
    query index and comma-separated reference indices, e.g.:

        query   ref
        0       0,1,2,3
        1       1,2,3,4

    python retrieve_touch.py \
        --ref_dir Taxim/results/gen_contact \
        --query_dir Taxim/results/gen_contact \
        --modality normal --scale 25 \
        --retrieval_mode tsv --tsv obj_36.tsv
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
# TSV loading
# ---------------------------------------------------------------------------

def load_tsv(tsv_path):
    """Parse a retrieval TSV file.

    Expected format (tab-separated, header required):
        query   ref
        0       0,1,2,3
        1       1,2,3,4

    Returns:
        List of (query_idx: int, ref_indices: list[int]) in file order.
    """
    results = []
    with open(tsv_path, newline="") as f:
        header = f.readline()   # consume header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            q_idx = int(parts[0])
            ref_idxs = [int(x) for x in parts[1].split(",")]
            results.append((q_idx, ref_idxs))
    return results


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

def compute_patch_mask(img_rgb, mask_mode, patch_size=14, img_size=224):
    """Compute a (num_patches,) boolean mask for DINOv2's forward_features.

    Pixels matching mask_mode are aggregated to patch level (any-pixel rule).
    True = masked patch (replaced by DINOv2's learned mask token).
    Returns None when mask_mode is 'none'.

    Args:
        img_rgb:    H×W×3 uint8 RGB array
        mask_mode:  'black_pixels' | 'white_pixels' | 'none'
        patch_size: ViT patch size in pixels (14 for all DINOv2 variants)
        img_size:   model input resolution (224)
    """
    if mask_mode == "none":
        return None

    if mask_mode == "black_pixels":
        pixel_mask = np.all(img_rgb == 0, axis=-1).astype(np.uint8)
    else:  # white_pixels
        pixel_mask = np.all(img_rgb == 255, axis=-1).astype(np.uint8)

    # Resize to model input size with nearest-neighbor to keep values binary
    mask_resized = cv2.resize(pixel_mask, (img_size, img_size),
                              interpolation=cv2.INTER_NEAREST)

    # Aggregate to patch grid: a patch is masked if any pixel inside it is masked
    n = img_size // patch_size   # 16 for 224/14
    patch_mask = mask_resized.reshape(n, patch_size, n, patch_size).any(axis=(1, 3))
    return torch.from_numpy(patch_mask.flatten())   # (n*n,) bool


def extract_features(model, transform, paths, device, batch_size=32, mask_mode="none"):
    """Extract DINOv2 CLS-token features for a list of image paths.

    Args:
        mask_mode: 'black_pixels' | 'white_pixels' | 'none'
                   When set, a patch-level boolean mask is derived from each image
                   and passed to model.forward_features(..., masks=masks) so that
                   DINOv2 replaces the flagged patch tokens with its learned mask
                   token before computing features.

    Returns:
        Tensor of shape (N, D), L2-normalised.
    """
    all_feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="  Extracting features", leave=False):
        batch_paths = paths[i:i + batch_size]
        imgs, patch_masks = [], []
        for p in batch_paths:
            img_bgr = cv2.imread(p)
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot read image: {p}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pm = compute_patch_mask(img_rgb, mask_mode)
            imgs.append(transform(img_rgb))
            if pm is not None:
                patch_masks.append(pm)

        batch = torch.stack(imgs).to(device)
        batch_masks = torch.stack(patch_masks).to(device) if patch_masks else None

        with torch.no_grad():
            out = model.forward_features(batch, masks=batch_masks)
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

def save_results(query_entries, ref_entries, topk_idxs, save_dir, topk_sims=None):
    """Save retrieval results as a .pkl file.

    The pkl contains a list of dicts, one per query, each with:
      - 'query_idx': int contact-point index in query folder
      - 'topk_ref_indices': list of int contact-point indices in ref folder (rank 1 first)
      - 'topk_similarities': list of float cosine similarities, or None if unavailable
    """
    results = []
    for qi, (q_idx, _) in enumerate(query_entries):
        row = {
            "query_idx": q_idx,
            "topk_ref_indices": [ref_entries[ri][0] for ri in topk_idxs[qi]],
            "topk_similarities": topk_sims[qi] if topk_sims is not None else None,
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

def make_figure(query_idx, query_path, ref_entries, topk_ref_list_idxs, save_dir,
                modality, k, topk_sims_list=None):
    """Save a single-row retrieval figure for one query entry.

    Layout: [Query | Rank-1 | Rank-2 | ... | Rank-K]

    Args:
        query_idx:          int, contact-point index of this query
        query_path:         str, path to query image
        ref_entries:        list of (ref_idx, ref_path) for the full reference set
        topk_ref_list_idxs: list of int — positions into ref_entries (not contact idx)
        save_dir:           str
        modality:           str
        k:                  int
        topk_sims_list:     list of float or None (no sim scores in TSV mode)
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
    for rank, list_idx in enumerate(topk_ref_list_idxs, start=1):
        ref_idx, ref_path = ref_entries[list_idx]
        if topk_sims_list is not None:
            sim_str = f" | sim={topk_sims_list[rank - 1]:.3f}"
        else:
            sim_str = ""
        axes[rank].imshow(read_rgb(ref_path))
        axes[rank].set_title(f"Ref #{ref_idx}\nrank {rank}{sim_str}", fontsize=9)
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
        description="Top-K touch retrieval using DINOv2 features or a pre-specified TSV."
    )
    parser.add_argument("--ref_dir", required=True, type=str,
                        help="Path to reference touch outputs folder.")
    parser.add_argument("--query_dir", required=True, type=str,
                        help="Path to query touch outputs folder.")
    parser.add_argument("--modality", required=True,
                        choices=["color", "normal", "curvature", "height"],
                        help="Static modality to use for indexing / visualization.")
    parser.add_argument("--scale", default=None, type=int,
                        help="Scale suffix in mm (e.g. 25 for _scale25_). "
                             "Omit to use base-resolution files.")
    parser.add_argument("--retrieval_mode", default="dinov2", choices=["dinov2", "tsv"],
                        help="Retrieval mode: 'dinov2' runs feature extraction; "
                             "'tsv' loads pre-specified results from --tsv.")
    # dinov2-mode args
    parser.add_argument("--top_k", default=5, type=int,
                        help="Number of top retrievals per query (dinov2 mode).")
    parser.add_argument("--mask_mode", default="none",
                        choices=["black_pixels", "white_pixels", "none"],
                        help="Pixels to mask during DINOv2 feature extraction. "
                             "Matching patches are replaced with DINOv2's learned "
                             "mask token. Default: none.")
    parser.add_argument("--dino_model", default="dinov2_vits14",
                        choices=["dinov2_vits14", "dinov2_vitb14",
                                 "dinov2_vitl14", "dinov2_vitg14"],
                        help="DINOv2 model variant (dinov2 mode).")
    # tsv-mode args
    parser.add_argument("--tsv", default=None, type=str,
                        help="Path to retrieval TSV file (tsv mode).")
    parser.add_argument("--save_dir", default="./log/touch_retrieval", type=str,
                        help="Output directory for results and figures.")
    args = parser.parse_args()

    if args.retrieval_mode == "tsv" and args.tsv is None:
        parser.error("--tsv is required when --retrieval_mode tsv is set.")

    os.makedirs(args.save_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Discover files (needed in both modes for image paths)
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
    # Build lookup: contact-point index → position in entries list
    # -----------------------------------------------------------------------
    ref_idx_to_pos = {idx: pos for pos, (idx, _) in enumerate(ref_entries)}
    query_idx_to_pos = {idx: pos for pos, (idx, _) in enumerate(query_entries)}

    # -----------------------------------------------------------------------
    # Branch on retrieval mode
    # -----------------------------------------------------------------------
    if args.retrieval_mode == "dinov2":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading DINOv2 model '{args.dino_model}' on {device}...")
        model, transform = load_dino_model(args.dino_model, device)

        ref_paths = [p for _, p in ref_entries]
        query_paths = [p for _, p in query_entries]

        print("Extracting reference features...")
        ref_feats = extract_features(model, transform, ref_paths, device,
                                     mask_mode=args.mask_mode)

        print("Extracting query features...")
        query_feats = extract_features(model, transform, query_paths, device,
                                       mask_mode=args.mask_mode)

        print(f"Computing top-{args.top_k} retrievals...")
        topk_idxs, topk_sims = compute_topk(query_feats, ref_feats, args.top_k)
        # topk_idxs: (N_q, K) — positions into ref_entries
        topk_idxs_list = topk_idxs.tolist()
        topk_sims_list = topk_sims.tolist()

        active_query_entries = query_entries
        k = args.top_k

    else:  # tsv
        print(f"Loading retrieval results from TSV: {args.tsv}")
        tsv_rows = load_tsv(args.tsv)

        # Resolve TSV contact-point indices to list positions; skip missing entries
        active_query_entries = []
        topk_idxs_list = []
        topk_sims_list = None   # TSV carries no similarity scores

        for q_idx, ref_contact_idxs in tsv_rows:
            if q_idx not in query_idx_to_pos:
                print(f"  Warning: query idx {q_idx} not found in query folder, skipping.")
                continue
            resolved_ref_positions = []
            for r_idx in ref_contact_idxs:
                if r_idx not in ref_idx_to_pos:
                    print(f"  Warning: ref idx {r_idx} not found in ref folder, skipping.")
                    continue
                resolved_ref_positions.append(ref_idx_to_pos[r_idx])

            q_pos = query_idx_to_pos[q_idx]
            active_query_entries.append(query_entries[q_pos])
            topk_idxs_list.append(resolved_ref_positions)

        k = max((len(r) for r in topk_idxs_list), default=0)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    save_results(
        query_entries=active_query_entries,
        ref_entries=ref_entries,
        topk_idxs=topk_idxs_list,
        save_dir=args.save_dir,
        topk_sims=topk_sims_list if args.retrieval_mode == "dinov2" else None,
    )

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    print("Generating retrieval figures...")
    for qi, (q_idx, q_path) in enumerate(tqdm(active_query_entries, desc="Figures")):
        list_idxs = topk_idxs_list[qi]
        sims = topk_sims_list[qi] if topk_sims_list is not None else None
        make_figure(
            query_idx=q_idx,
            query_path=q_path,
            ref_entries=ref_entries,
            topk_ref_list_idxs=list_idxs,
            save_dir=args.save_dir,
            modality=args.modality,
            k=k,
            topk_sims_list=sims,
        )

    print(f"Done. Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
