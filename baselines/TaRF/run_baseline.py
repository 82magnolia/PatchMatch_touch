#!/usr/bin/env python3
"""Deterministic one-shot TaRF baseline for PatchMatch_touch."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_ROOT.parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

from patchmatch_tarf.conditions import resolve_conditions
from patchmatch_tarf.contracts import (
    RETRIEVAL_MODES,
    VIDEO_TYPES,
    discover_indices,
    output_names,
    resolve_pairs,
)
from patchmatch_tarf.generator import SmokeGenerator, TaRFGenerator, validate_tarf_assets
from patchmatch_tarf.media import require_cv2, video_info, write_repeated_video
from patchmatch_tarf.view_renderer import prepare_fixed_view_conditions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref_dir", required=True, type=Path)
    parser.add_argument("--query_dir", required=True, type=Path)
    parser.add_argument("--save_dir", required=True, type=Path)
    parser.add_argument("--scale", nargs="+", type=float, default=[100.0])
    parser.add_argument("--video_type", choices=VIDEO_TYPES, default="shadow")
    parser.add_argument(
        "--retrieval_mode", choices=RETRIEVAL_MODES, default="sim_gt_retrieval"
    )
    parser.add_argument("--tsv", type=Path)
    parser.add_argument("--retrieval_modality", default="normal")
    parser.add_argument("--dino_weights", type=Path)
    parser.add_argument("--query_indices", nargs="+", type=int)

    parser.add_argument(
        "--conditions_dir",
        type=Path,
        help="Optional NeRF RGB/depth root; defaults to --query_dir static modalities.",
    )
    parser.add_argument(
        "--condition_manifest",
        type=Path,
        help='JSON mapping query IDs to {"rgb": [...], "depth": [...]} paths.',
    )
    parser.add_argument(
        "--condition_geometry",
        choices=("auto", "fixed_views", "files"),
        default="auto",
        help=(
            "Build original-TaRF fixed-standoff views, or consume existing files. "
            "Auto builds views unless --conditions_dir/--condition_manifest is given."
        ),
    )
    parser.add_argument(
        "--sensor_offset_file",
        type=Path,
        default=PROJECT_ROOT / "log" / "gelsight_sensor_offset.json",
        help=(
            "GelSight marker-to-gel calibration JSON produced by "
            "real_data_transfer/calibrate_sensor_offset.py."
        ),
    )
    parser.add_argument("--condition_render_size", type=int, default=480)
    parser.add_argument("--background", required=True, type=Path)
    parser.add_argument(
        "--timing_suffix",
        default="render_mask",
        help="Non-tactile query video suffix used only for frame count/resolution/FPS.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=BASELINE_ROOT / "img2touch" / "configs" / "tarf.yaml",
    )
    parser.add_argument(
        "--diffusion_ckpt",
        type=Path,
        default=BASELINE_ROOT / "img2touch" / "pretrained_models" / "img2touch.ckpt",
    )
    parser.add_argument("--first_stage_ckpt", type=Path)
    parser.add_argument(
        "--ranking_rgb_enc_ckpt",
        type=Path,
        default=BASELINE_ROOT
        / "img2touch"
        / "pretrained_models"
        / "reranking_rgb_enc.ckpt",
    )
    parser.add_argument(
        "--ranking_tac_enc_ckpt",
        type=Path,
        default=BASELINE_ROOT
        / "img2touch"
        / "pretrained_models"
        / "reranking_tac_enc.ckpt",
    )
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--depth_multiplier", type=float, default=1.0)
    parser.add_argument("--depth_clip_max", type=float, default=5.0)
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Use a condition-only procedural generator to test plumbing; not a TaRF result.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--debug_images", action="store_true")
    return parser


def run_dinov3(args, retrieval_dir: Path) -> list[tuple[int, int]]:
    if args.dino_weights is None:
        raise ValueError("--dino_weights is required for retrieval_mode=dinov3")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "retrieve_touch.py"),
        "--ref_dir",
        str(args.ref_dir),
        "--query_dir",
        str(args.query_dir),
        "--modality",
        args.retrieval_modality,
        "--scale",
        f"{args.scale[0]:g}",
        "--retrieval_mode",
        "dinov3",
        "--top_k",
        "1",
        "--dinov3_weights",
        str(args.dino_weights),
        "--save_dir",
        str(retrieval_dir),
        "--no_figures",
    ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    with (retrieval_dir / "results.pkl").open("rb") as stream:
        rows = pickle.load(stream)
    return [
        (int(row["query_idx"]), int(row["topk_ref_indices"][0]))
        for row in rows
        if row["topk_ref_indices"]
    ]


def save_pairing(save_dir: Path, pairs: list[tuple[int, int]], mode: str) -> None:
    retrieval_dir = save_dir / "retrieval"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "query_idx": query,
            "topk_ref_indices": [reference],
            "topk_similarities": None,
        }
        for query, reference in pairs
    ]
    with (retrieval_dir / "results.pkl").open("wb") as stream:
        pickle.dump(rows, stream)
    if mode in ("sim_gt_retrieval", "real_gt_retrieval"):
        mapping = save_dir / (
            "odd_to_even.tsv" if mode == "real_gt_retrieval" else "identity.tsv"
        )
        mapping.write_text(
            "query\tref\n"
            + "".join(f"{query}\t{reference}\n" for query, reference in pairs)
        )


def json_ready(args, pairs, condition_records, view_records) -> dict:
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "pairs": pairs,
            "conditions": condition_records,
            "fixed_view_geometry": view_records,
            "generator": "smoke_test_not_tarf" if args.smoke_test else "tarf_diffusion",
            "query_tactile_used_for_prediction": False,
        }
    )
    return config


def main() -> None:
    args = build_parser().parse_args()
    for key in ("ref_dir", "query_dir", "save_dir", "background", "config"):
        setattr(args, key, getattr(args, key).resolve())
    for key in (
        "conditions_dir",
        "condition_manifest",
        "tsv",
        "dino_weights",
        "diffusion_ckpt",
        "first_stage_ckpt",
        "ranking_rgb_enc_ckpt",
        "ranking_tac_enc_ckpt",
        "sensor_offset_file",
    ):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    if not args.ref_dir.is_dir() or not args.query_dir.is_dir():
        raise SystemExit(
            f"Reference/query directories must exist: {args.ref_dir}, {args.query_dir}"
        )
    if args.depth_clip_max <= 0 or args.depth_multiplier <= 0:
        raise SystemExit("--depth_clip_max and --depth_multiplier must be positive")

    ref_indices = discover_indices(
        args.ref_dir, args.scale[0], args.retrieval_modality
    )
    query_indices = discover_indices(
        args.query_dir, args.scale[0], args.retrieval_modality
    )
    retrieval_dir = args.save_dir / "retrieval"
    if args.retrieval_mode == "dinov3":
        pairs = [] if args.dry_run else run_dinov3(args, retrieval_dir)
    else:
        pairs = resolve_pairs(
            args.retrieval_mode, ref_indices, query_indices, args.tsv
        )
    if args.query_indices is not None and pairs:
        requested = set(args.query_indices)
        pairs = [pair for pair in pairs if pair[0] in requested]
        absent = sorted(requested - {query for query, _ in pairs})
        if absent:
            raise SystemExit(f"Requested query indices are unavailable: {absent}")
    if not pairs and not (args.dry_run and args.retrieval_mode == "dinov3"):
        raise SystemExit("No query/reference pairs were resolved")

    conditions = {}
    condition_records = {}
    view_records = {}
    geometry_mode = args.condition_geometry
    if geometry_mode == "auto":
        geometry_mode = (
            "files"
            if args.conditions_dir is not None or args.condition_manifest is not None
            else "fixed_views"
        )
    if geometry_mode == "fixed_views" and args.condition_manifest is not None:
        raise SystemExit("--condition_manifest cannot be combined with fixed-view rendering")
    resolved_conditions_dir = args.conditions_dir
    if geometry_mode == "fixed_views":
        resolved_conditions_dir = args.save_dir / "conditions"
    for query_idx, _ in pairs:
        try:
            if geometry_mode == "fixed_views":
                _, view_record = prepare_fixed_view_conditions(
                    args.query_dir,
                    query_idx,
                    resolved_conditions_dir,
                    sensor_offset_file=args.sensor_offset_file,
                    size=args.condition_render_size,
                )
                view_records[str(query_idx)] = view_record
            resolved = resolve_conditions(
                query_dir=args.query_dir,
                query_idx=query_idx,
                scale=args.scale[0],
                background_path=args.background,
                conditions_dir=resolved_conditions_dir,
                manifest=args.condition_manifest,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        conditions[query_idx] = resolved
        condition_records[str(query_idx)] = resolved.as_dict()

    resolved_config = json_ready(args, pairs, condition_records, view_records)
    if args.dry_run:
        if not args.smoke_test:
            try:
                validate_tarf_assets(
                    args.config,
                    args.diffusion_ckpt,
                    args.first_stage_ckpt,
                    args.ranking_rgb_enc_ckpt,
                    args.ranking_tac_enc_ckpt,
                )
            except FileNotFoundError as exc:
                raise SystemExit(str(exc)) from exc
        print(json.dumps(resolved_config, indent=2, sort_keys=True))
        return

    args.save_dir.mkdir(parents=True, exist_ok=True)
    save_pairing(args.save_dir, pairs, args.retrieval_mode)
    (args.save_dir / "resolved_config.json").write_text(
        json.dumps(resolved_config, indent=2, sort_keys=True)
    )
    if args.smoke_test:
        generator = SmokeGenerator(n_samples=args.n_samples, seed=args.seed)
    else:
        try:
            generator = TaRFGenerator(
                source_root=BASELINE_ROOT,
                config=args.config,
                diffusion_ckpt=args.diffusion_ckpt,
                first_stage_ckpt=args.first_stage_ckpt,
                ranking_rgb_ckpt=args.ranking_rgb_enc_ckpt,
                ranking_tac_ckpt=args.ranking_tac_enc_ckpt,
                n_samples=args.n_samples,
                ddim_steps=args.ddim_steps,
                guidance_scale=args.guidance_scale,
                ddim_eta=args.ddim_eta,
                seed=args.seed,
                device=args.device,
                depth_multiplier=args.depth_multiplier,
                depth_clip_max=args.depth_clip_max,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc

    cv2 = require_cv2()
    transfer_dir = args.save_dir / "transfer"
    generation_dir = args.save_dir / "generation"
    transfer_dir.mkdir(exist_ok=True)
    generation_dir.mkdir(exist_ok=True)
    generation_metadata = {}
    for query_idx, reference_idx in pairs:
        timing_video = args.query_dir / f"{query_idx}_{args.timing_suffix}.mp4"
        reference_video = args.ref_dir / f"{reference_idx}_{args.video_type}.mp4"
        query_video = args.query_dir / f"{query_idx}_{args.video_type}.mp4"
        for path, label in (
            (timing_video, "non-tactile timing video"),
            (reference_video, "reference tactile video"),
            (query_video, "query tactile video copied/evaluated only after prediction"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")
        frame_count, width, height, fps = video_info(timing_video)
        selected_rgb, candidates_rgb, scores, selected = generator.generate(
            conditions[query_idx], query_idx
        )
        query_generation_dir = generation_dir / str(query_idx)
        query_generation_dir.mkdir(exist_ok=True)
        for candidate_idx, candidate_rgb in enumerate(candidates_rgb):
            cv2.imwrite(
                str(query_generation_dir / f"candidate_{candidate_idx:02}.png"),
                cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2BGR),
            )
        selected_bgr = cv2.cvtColor(selected_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(query_generation_dir / "selected.png"), selected_bgr)
        names = output_names(query_idx, args.video_type)
        write_repeated_video(
            selected_bgr,
            transfer_dir / names["prediction"],
            frame_count,
            width,
            height,
            fps,
        )
        shutil.copy2(reference_video, transfer_dir / names["reference"])
        shutil.copy2(query_video, transfer_dir / names["query"])
        generation_metadata[str(query_idx)] = {
            "selected_candidate": selected,
            "ranking_scores": scores,
            "timing_source": str(timing_video),
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "fps": fps,
        }
        if args.debug_images:
            shutil.copy2(
                conditions[query_idx].rgb_paths[0],
                query_generation_dir / f"condition_rgb{conditions[query_idx].rgb_paths[0].suffix}",
            )
    (generation_dir / "metadata.json").write_text(
        json.dumps(generation_metadata, indent=2, sort_keys=True)
    )

    if not args.skip_eval:
        sys.path.insert(0, str(PROJECT_ROOT))
        try:
            from transfer_pipeline import _evaluate_videos
        except ImportError as exc:
            raise SystemExit(
                "Evaluation needs torch, lpips, and scikit-image; install environment.yml "
                "or pass --skip_eval."
            ) from exc
        _evaluate_videos(
            pred_dir=str(transfer_dir),
            query_dir=str(args.query_dir),
            video_type=args.video_type,
            pred_glob="*_transferred.mp4",
            query_stem_fn=lambda index: f"{index}_{args.video_type}.mp4",
            out_pkl=str(transfer_dir / "metrics.pkl"),
        )


if __name__ == "__main__":
    main()
