#!/usr/bin/env python3
"""Extract only TaRF's frozen autoencoder weights from img2touch.ckpt."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.source, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    prefix = "first_stage_model."
    first_stage = {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not first_stage:
        raise RuntimeError(f"{args.source} has no {prefix} weights")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": first_stage}, args.output)
    print(f"Saved {len(first_stage)} first-stage tensors to {args.output}")


if __name__ == "__main__":
    main()
