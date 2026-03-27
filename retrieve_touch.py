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

def make_figure(query_idx, query_paths_by_mod, ref_entries_by_mod, topk_ref_list_idxs,
                save_dir, modalities, k, topk_sims_list=None):
    """Save a retrieval figure for one query entry.

    Grid layout: rows = modalities, columns = [Query | Rank-1 | ... | Rank-K]
    Modality labels appear on the left; entry labels appear on the top row only.

    Args:
        query_idx:           int, contact-point index of this query
        query_paths_by_mod:  dict {modality: path} for this query
        ref_entries_by_mod:  dict {modality: [(ref_idx, ref_path), ...]} (full ref set,
                             aligned to the same contact-point order across modalities)
        topk_ref_list_idxs:  list of int — positions into the ref lists
        save_dir:            str
        modalities:          list of str
        k:                   int
        topk_sims_list:      list of float or None (TSV mode has no scores)
    """
    M = len(modalities)
    n_cols = 1 + len(topk_ref_list_idxs)
    fig, axes = plt.subplots(M, n_cols,
                             figsize=(3 * n_cols, 3 * M + 0.5),
                             squeeze=False)

    def read_rgb(path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for row, mod in enumerate(modalities):
        ref_ents_mod = ref_entries_by_mod[mod]

        # Query column
        ax = axes[row, 0]
        ax.imshow(read_rgb(query_paths_by_mod[mod]))
        if row == 0:
            ax.set_title(f"Query #{query_idx}", fontsize=9, fontweight="bold")
        # Modality row label on the left edge
        ax.text(-0.05, 0.5, mod, transform=ax.transAxes,
                ha="right", va="center", rotation=90, fontsize=9)
        ax.axis("off")

        # Retrieved columns
        for rank, list_idx in enumerate(topk_ref_list_idxs, start=1):
            ref_idx, ref_path = ref_ents_mod[list_idx]
            ax = axes[row, rank]
            ax.imshow(read_rgb(ref_path))
            if row == 0:
                sim_str = (f"\nsim={topk_sims_list[rank - 1]:.3f}"
                           if topk_sims_list is not None else "")
                ax.set_title(f"Ref #{ref_idx} (rank {rank}){sim_str}", fontsize=9)
            ax.axis("off")

    fig.suptitle(
        f"Query #{query_idx} — Top-{k} retrieval ({', '.join(modalities)})",
        fontsize=10,
    )
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
    parser.add_argument("--modality", required=True, nargs='+',
                        choices=["color", "normal", "curvature", "height", "shapeindex"],
                        help="Modality(ies) to use for indexing. If multiple are given, "
                             "DINOv2 features are extracted independently per modality and "
                             "concatenated. The first modality is used for visualization.")
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

    def _discover_all_modalities(folder, modalities, scale):
        """Discover files for each modality; return (by_mod, common_idxs).

        by_mod:       dict {modality -> {idx: path}}
        common_idxs:  sorted list of contact-point indices present in all modalities.
        """
        by_mod = {}
        for mod in modalities:
            entries = discover_files(folder, mod, scale)
            if not entries:
                scale_str = f"_scale{scale}_" if scale else "_"
                raise FileNotFoundError(
                    f"No files matching '{{}}{scale_str}{mod}.jpg' found in {folder}"
                )
            by_mod[mod] = {idx: p for idx, p in entries}
        common_idxs = sorted(set.intersection(*[set(d) for d in by_mod.values()]))
        return by_mod, common_idxs

    def _build_entries_by_mod(by_mod, common_idxs, modalities):
        """Build {modality -> [(idx, path), ...]} aligned to common_idxs."""
        return {mod: [(idx, by_mod[mod][idx]) for idx in common_idxs]
                for mod in modalities}

    # -----------------------------------------------------------------------
    # Discover files for all modalities (both modes need them for figures)
    # -----------------------------------------------------------------------
    print(f"Discovering reference files ({', '.join(args.modality)}) in: {args.ref_dir}")
    ref_by_mod, common_ref_idxs = _discover_all_modalities(
        args.ref_dir, args.modality, args.scale)
    ref_entries_by_mod = _build_entries_by_mod(ref_by_mod, common_ref_idxs, args.modality)
    print(f"  Found {len(common_ref_idxs)} reference entries.")

    print(f"Discovering query files ({', '.join(args.modality)}) in: {args.query_dir}")
    query_by_mod, common_query_idxs = _discover_all_modalities(
        args.query_dir, args.modality, args.scale)
    query_entries_by_mod = _build_entries_by_mod(query_by_mod, common_query_idxs, args.modality)
    print(f"  Found {len(common_query_idxs)} query entries.")

    # Canonical (single-modality) entry lists used for save_results and index lookups
    vis_modality  = args.modality[0]
    ref_entries   = ref_entries_by_mod[vis_modality]
    query_entries = query_entries_by_mod[vis_modality]

    # -----------------------------------------------------------------------
    # Branch on retrieval mode
    # -----------------------------------------------------------------------
    if args.retrieval_mode == "dinov2":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading DINOv2 model '{args.dino_model}' on {device}...")
        model, transform = load_dino_model(args.dino_model, device)

        # Extract features per modality, concatenate, re-normalise
        ref_feats_list, query_feats_list = [], []
        for mod in args.modality:
            ref_paths_mod   = [ref_by_mod[mod][idx]   for idx in common_ref_idxs]
            query_paths_mod = [query_by_mod[mod][idx] for idx in common_query_idxs]
            print(f"Extracting reference features ({mod})...")
            ref_feats_list.append(
                extract_features(model, transform, ref_paths_mod, device,
                                 mask_mode=args.mask_mode))
            print(f"Extracting query features ({mod})...")
            query_feats_list.append(
                extract_features(model, transform, query_paths_mod, device,
                                 mask_mode=args.mask_mode))

        ref_feats   = F.normalize(torch.cat(ref_feats_list,   dim=-1), dim=-1)
        query_feats = F.normalize(torch.cat(query_feats_list, dim=-1), dim=-1)

        print(f"Computing top-{args.top_k} retrievals...")
        topk_idxs, topk_sims = compute_topk(query_feats, ref_feats, args.top_k)
        topk_idxs_list  = topk_idxs.tolist()
        topk_sims_list  = topk_sims.tolist()
        active_query_entries_by_mod = query_entries_by_mod
        k = args.top_k

    else:  # tsv
        print(f"Loading retrieval results from TSV: {args.tsv}")
        tsv_rows = load_tsv(args.tsv)

        ref_idx_to_pos   = {idx: pos for pos, (idx, _) in enumerate(ref_entries)}
        query_idx_to_pos = {idx: pos for pos, (idx, _) in enumerate(query_entries)}

        active_query_positions = []   # positions into query_entries_by_mod lists
        topk_idxs_list = []
        topk_sims_list = None   # TSV carries no similarity scores

        for q_idx, ref_contact_idxs in tsv_rows:
            if q_idx not in query_idx_to_pos:
                print(f"  Warning: query idx {q_idx} not found in query folder, skipping.")
                continue
            resolved = []
            for r_idx in ref_contact_idxs:
                if r_idx not in ref_idx_to_pos:
                    print(f"  Warning: ref idx {r_idx} not found in ref folder, skipping.")
                    continue
                resolved.append(ref_idx_to_pos[r_idx])
            active_query_positions.append(query_idx_to_pos[q_idx])
            topk_idxs_list.append(resolved)

        active_query_entries_by_mod = {
            mod: [query_entries_by_mod[mod][pos] for pos in active_query_positions]
            for mod in args.modality
        }
        k = max((len(r) for r in topk_idxs_list), default=0)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    active_query_entries = active_query_entries_by_mod[vis_modality]
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
    for qi, (q_idx, _) in enumerate(tqdm(active_query_entries, desc="Figures")):
        query_paths_by_mod = {mod: active_query_entries_by_mod[mod][qi][1]
                              for mod in args.modality}
        sims = topk_sims_list[qi] if topk_sims_list is not None else None
        make_figure(
            query_idx=q_idx,
            query_paths_by_mod=query_paths_by_mod,
            ref_entries_by_mod=ref_entries_by_mod,
            topk_ref_list_idxs=topk_idxs_list[qi],
            save_dir=args.save_dir,
            modalities=args.modality,
            k=k,
            topk_sims_list=sims,
        )

    print(f"Done. Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
