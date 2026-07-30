"""Shared helpers for query-conditioning flags, used by train/finetune/eval.

Two independent, opt-in conditioning signals (see models/archs.py):
  --mask_cond       concat the per-frame render_mask (1ch, sensor-aligned) to lq
  --film_modality   a static geometry render (normal/curvature/height, 3ch) fed
                    to the model as global FiLM (alignment-agnostic)

Both are query-side signals available at deployment. Do NOT condition on the
tactile-video normals ({idx}_tactile_normal.mp4) — those are derived from the GT.

The conditioned model must be built with the SAME cond dims in sim pretraining
and real fine-tuning so checkpoints load without any weight surgery.
"""

FILM_CHANS = 3   # normal/curvature/height renders are 3-channel jpgs
MASK_CHANS = 1   # render_mask reduced to a single channel


def add_cond_args(p):
    p.add_argument('--cond_dir', default=None,
                   help="Root of query conditioning signals: {cond_dir}/{obj}/"
                        "{pair}_render_mask.mp4 and {pair}_scale{S}_{modality}.jpg "
                        "(sim: gen_contact_full_query_*; real: log/real_data_gt_retrieval)")
    p.add_argument('--mask_cond', action='store_true',
                   help="Concat the per-frame render_mask as an aligned input channel")
    p.add_argument('--film_modality', default='none',
                   choices=['none', 'normal', 'curvature', 'height'],
                   help="Static geometry render injected via global FiLM")
    p.add_argument('--film_scale', type=int, default=4,
                   help="Scale suffix of the FiLM geometry jpg (sim uses 100, real 4)")
    return p


def film_modality(args):
    m = getattr(args, 'film_modality', 'none')
    return None if m in (None, 'none') else m


def cond_dims(args):
    """(cond_chans, film_chans) for build_model."""
    cond_chans = MASK_CHANS if getattr(args, 'mask_cond', False) else 0
    film_chans = FILM_CHANS if film_modality(args) else 0
    return cond_chans, film_chans


def dataset_cond_kwargs(args):
    """kwargs for TactileTransferDataset / RealTactileTransferDataset."""
    return dict(cond_dir=getattr(args, 'cond_dir', None),
                mask_cond=getattr(args, 'mask_cond', False),
                film_modality=film_modality(args),
                film_scale=getattr(args, 'film_scale', 4))


def check_cond_args(args):
    """Validate that a cond_dir is given when any conditioning is requested."""
    if (getattr(args, 'mask_cond', False) or film_modality(args)) and not getattr(args, 'cond_dir', None):
        raise SystemExit("--cond_dir is required when --mask_cond or --film_modality is set")
