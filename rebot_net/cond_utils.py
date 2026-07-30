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
    p.add_argument('--time_cond', default='none',
                   choices=['none', 'film', 'token', 'film_token', 'concat'],
                   help="Condition on the frame's normalized timestamp (index/(n-1) in "
                        "[0,1], where a touch runs no-press -> contact -> take-off). "
                        "'film': sinusoidal timestep embedding -> MLP -> per-ConvNeXt-stage "
                        "FiLM (scale/shift), the Stable-Diffusion/DDPM recipe. 'token': the "
                        "same embedding projected to one bias vector added to every "
                        "bottleneck token. 'film_token': both injection points. 'concat': "
                        "broadcast the scalar time as one extra constant input channel per "
                        "frame (coord-conv style). 'none' (default) leaves the net unchanged.")
    return p


def film_modality(args):
    m = getattr(args, 'film_modality', 'none')
    return None if m in (None, 'none') else m


def time_cond_mode(args):
    """Full --time_cond value ('none'/'film'/'token'/'film_token'/'concat')."""
    return getattr(args, 'time_cond', 'none')


def time_concat(args):
    """True when time is injected as an extra constant input channel."""
    return time_cond_mode(args) == 'concat'


def time_module_mode(args):
    """In-network TimeConditioner mode ('film'/'token'/'film_token'), or None.

    'concat' and 'none' need no in-network module -- 'concat' rides the generic
    per-frame cond-channel path, 'none' is a no-op.
    """
    m = time_cond_mode(args)
    return m if m in ('film', 'token', 'film_token') else None


def uses_time(args):
    """True when the model must receive the scalar timestamp t (module modes)."""
    return time_module_mode(args) is not None


def cond_dims(args):
    """(cond_chans, film_chans) for build_model.

    cond_chans counts every per-frame channel appended to the RGB input: the
    render_mask (mask_cond) and, for --time_cond concat, one constant time
    channel. These ride the same generic cond-channel path in the network.
    """
    cond_chans = (MASK_CHANS if getattr(args, 'mask_cond', False) else 0) \
        + (1 if time_concat(args) else 0)
    film_chans = FILM_CHANS if film_modality(args) else 0
    return cond_chans, film_chans


def dataset_cond_kwargs(args):
    """kwargs for TactileTransferDataset / RealTactileTransferDataset."""
    return dict(cond_dir=getattr(args, 'cond_dir', None),
                mask_cond=getattr(args, 'mask_cond', False),
                film_modality=film_modality(args),
                film_scale=getattr(args, 'film_scale', 4),
                time_cond=time_cond_mode(args))


def check_cond_args(args):
    """Validate that a cond_dir is given when any conditioning is requested."""
    if (getattr(args, 'mask_cond', False) or film_modality(args)) and not getattr(args, 'cond_dir', None):
        raise SystemExit("--cond_dir is required when --mask_cond or --film_modality is set")
