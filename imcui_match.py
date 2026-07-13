"""Sparse/dense feature matching via image-matching-webui (IMCUI), bypassing
its Gradio-only imcui.ui.utils / imcui.api.core.ImageMatchingAPI (which
unconditionally import gradio/spaces/datasets/poselib just to be imported at
all -- not needed for a headless single-pair matching call).

Instead this calls IMCUI's own hloc.extract_features / hloc.match_features /
hloc.match_dense directly (the same functions ImageMatchingAPI itself calls
internally), and feeds the resulting raw matched keypoints into
dinov3.dense_match._fit_dense_field for the RANSAC-inlier-selection +
affine/homography/rbf geometric fit -- the same fitting stage the DINOv3
backend uses -- so every matcher backend in this repo shares one consistent
NNF-fitting implementation and CLI surface (--*_reproj_threshold,
--*_transform_type).

Supported methods (--matcher choices in main_retrieval_transfer_feat_match.py):
  disk_lightglue        DISK keypoints + LightGlue matcher
  superpoint_superglue  SuperPoint keypoints + SuperGlue matcher
  loftr                 LoFTR (end-to-end dense matcher, no separate detector)
  superpoint_lightglue  SuperPoint keypoints + LightGlue matcher
  sift_lightglue        SIFT keypoints + LightGlue matcher

All five auto-download their pretrained weights on first use (HuggingFace
Hub `Realcat/imcui_checkpoints` for SuperPoint/SuperGlue/LightGlue weights;
kornia's own hub cache for DISK/LoFTR) -- no manual weights path needed,
unlike DINOv3's gated checkpoint.

Setup required once before any of these will import successfully:
  pip install kornia omegaconf h5py
  pip install git+https://github.com/cvg/LightGlue.git
  git clone https://github.com/magicleap/SuperGluePretrainedNetwork \
      image-matching-webui/imcui/third_party/SuperGluePretrainedNetwork
(SuperGluePretrainedNetwork provides the SuperPoint extractor too, so it's
required for superpoint_superglue AND superpoint_lightglue, not just SuperGlue.)
"""
import os
import sys

import numpy as np
import torch
import yaml

_IMCUI_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image-matching-webui")
if _IMCUI_ROOT not in sys.path:
    sys.path.insert(0, _IMCUI_ROOT)

from imcui.hloc import extract_features, match_features, match_dense
from imcui.hloc import extractors as _extractors_pkg
from imcui.hloc import matchers as _matchers_pkg
from imcui.hloc.utils.base_model import dynamic_load

_APP_YAML = os.path.join(_IMCUI_ROOT, "config", "app.yaml")

METHOD_TO_ZOO_KEY = {
    "disk_lightglue": "disk+lightglue",
    "superpoint_superglue": "superpoint+superglue",
    "loftr": "loftr",
    "superpoint_lightglue": "superpoint+lightglue",
    "sift_lightglue": "sift+lightglue",
}

_model_cache = {}   # method -> (extractor_or_None, matcher, conf)
_zoo_cache = None


def _get_device():
    # Must match hloc.extract_features.extract()'s own internal device
    # resolution exactly (it hardcodes this, ignoring any device we'd pass
    # in) -- otherwise the model (placed on our choice) and its input tensor
    # (placed on extract()'s choice) end up on different devices.
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_zoo():
    global _zoo_cache
    if _zoo_cache is None:
        with open(_APP_YAML) as f:
            cfg = yaml.safe_load(f)
        _zoo_cache = cfg["matcher_zoo"]
    return _zoo_cache


def _parse_conf(zoo_entry):
    """Mirrors imcui.ui.utils.parse_match_config, without importing that
    (gradio/spaces/datasets-heavy) module."""
    if zoo_entry["standalone"]:
        return {
            "matcher": match_dense.confs.get(zoo_entry["matcher"]),
            "standalone": True,
        }
    return {
        "feature": extract_features.confs.get(zoo_entry["feature"]),
        "matcher": match_features.confs.get(zoo_entry["matcher"]),
        "standalone": False,
    }


def _get_model(module_pkg, conf):
    """Mirrors imcui.ui.utils.get_model / get_feature_model (dynamic_load +
    instantiate + eval), without needing that module's gradio-heavy imports.
    """
    Model = dynamic_load(module_pkg, conf["model"]["name"])
    return Model(conf["model"]).eval()


def _load_method(method):
    if method in _model_cache:
        return _model_cache[method]
    if method not in METHOD_TO_ZOO_KEY:
        raise ValueError(f"Unknown IMCUI method: {method!r}; choices: {list(METHOD_TO_ZOO_KEY)}")

    device = _get_device()
    zoo = _load_zoo()
    conf = _parse_conf(zoo[METHOD_TO_ZOO_KEY[method]])

    matcher = _get_model(_matchers_pkg, conf["matcher"]).to(device)
    extractor = None
    if not conf["standalone"]:
        extractor = _get_model(_extractors_pkg, conf["feature"]).to(device)

    _model_cache[method] = (extractor, matcher, conf)
    return _model_cache[method]


def compute_imcui_sparse_matches(image_left, image_right, method):
    """Run one of IMCUI's matchers on a (ref, query) image pair.

    image_left / image_right: float32 (H, W, 3) arrays in [0, 1] -- same
    convention as dinov3.dense_match.compute_dinov3_nnf (left=REF, sampled;
    right=QUERY, output grid).

    Returns (pts_l, pts_r): (row, col) matched-point arrays in each image's
    original pixel space, mirroring dinov3.dense_match._find_sparse_matches's
    return contract, so they can be fed directly into
    dinov3.dense_match._fit_dense_field.
    """
    extractor, matcher, conf = _load_method(method)

    img_l_u8 = (np.clip(image_left, 0, 1) * 255).astype(np.uint8)
    img_r_u8 = (np.clip(image_right, 0, 1) * 255).astype(np.uint8)

    try:
        if conf["standalone"]:
            pred = match_dense.match_images(
                matcher, img_l_u8, img_r_u8, conf["matcher"]["preprocessing"], device=_get_device())
        else:
            feat_l = extract_features.extract(extractor, img_l_u8, conf["feature"]["preprocessing"])
            feat_r = extract_features.extract(extractor, img_r_u8, conf["feature"]["preprocessing"])
            pred = match_features.match_images(matcher, feat_l, feat_r)
    except IndexError as e:
        # image-matching-webui's match_features.match_images crashes on this
        # exact exception when zero keypoints survive matching (mkpts0/1
        # come back as malformed 1-D empty arrays instead of (0, 2), which
        # breaks its internal scale_keypoints call) -- translate to
        # RuntimeError so callers (e.g. main_retrieval_transfer_feat_match.py)
        # can fall back to an identity NNF the same way they already do for
        # "too few matches" from the DINOv3 backend.
        raise RuntimeError(
            f"IMCUI matcher {method!r} likely found zero matches between the "
            f"image pair (underlying error: {e}).") from e

    mkpts_l_xy = pred["mkeypoints0_orig"]
    mkpts_r_xy = pred["mkeypoints1_orig"]
    if len(mkpts_l_xy) == 0:
        raise RuntimeError(f"IMCUI matcher {method!r} found zero matches between the image pair.")

    pts_l = mkpts_l_xy[:, ::-1]   # (x, y) -> (row, col)
    pts_r = mkpts_r_xy[:, ::-1]
    return pts_l.astype(np.float64), pts_r.astype(np.float64)


def compute_imcui_nnf(image_left, image_right, method,
                      reproj_threshold, transform_type):
    """Dense correspondence NNF via an IMCUI sparse/dense matcher + the same
    RANSAC-inlier-selection/geometric-fit stage dinov3.dense_match uses.
    Mirrors dinov3.dense_match.compute_dinov3_nnf's contract exactly (same
    left/right convention, same return: (H_right, W_right, 2) int32 NNF).
    """
    pts_l, pts_r = compute_imcui_sparse_matches(image_left, image_right, method)

    from dinov3.dense_match import _fit_dense_field
    h2, w2 = image_right.shape[:2]
    src_row, src_col, inlier_count, total = _fit_dense_field(
        pts_l, pts_r, h2, w2, transform_type, reproj_threshold)

    nnf = np.zeros((h2, w2, 2), dtype=np.int32)
    nnf[..., 0] = np.clip(np.round(src_col), 0, image_left.shape[1] - 1)
    nnf[..., 1] = np.clip(np.round(src_row), 0, image_left.shape[0] - 1)
    return nnf
