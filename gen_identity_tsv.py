"""
Generate an identity-mapping TSV for retrieve_touch.py (tsv mode).

Each object subdirectory in gen_contact_full/<obj>/ and
gen_contact_full_query/<obj>/ shares the same touch-location indices by
design, so a single identity mapping (query idx i -> ref idx i) is valid
for every object.

The script discovers the indices from one sample object subdirectory and
writes a TSV to --save_path with the format expected by retrieve_touch.py:

    query   ref
    0       0
    1       1
    ...

Usage:
    python gen_identity_tsv.py
    python gen_identity_tsv.py --ref_root Taxim/results/gen_contact_full --scale 100 --save_path log/identity_mapping.tsv
"""

import argparse
import os
import re


def discover_touch_indices(folder, scale):
    pattern = re.compile(rf"^(\d+)_scale{scale}_normal\.jpg$")
    indices = []
    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


def main():
    parser = argparse.ArgumentParser(
        description="Generate identity-mapping TSV for retrieve_touch.py."
    )
    parser.add_argument("--ref_root", default="Taxim/results/gen_contact_full",
                        help="Root reference directory containing per-object subdirs.")
    parser.add_argument("--scale", default=100, type=int,
                        help="Scale suffix used in filenames (default: 100).")
    parser.add_argument("--sample_obj", default=None, type=str,
                        help="Object subdir to sample indices from. "
                             "Defaults to the numerically first subdir found.")
    parser.add_argument("--save_path", default="log/identity_mapping.tsv",
                        help="Output TSV path (default: log/identity_mapping.tsv).")
    args = parser.parse_args()

    # Pick a sample object subdirectory
    if args.sample_obj is None:
        subdirs = sorted(
            [d for d in os.listdir(args.ref_root)
             if os.path.isdir(os.path.join(args.ref_root, d))],
            key=lambda x: int(x) if x.isdigit() else x,
        )
        if not subdirs:
            raise FileNotFoundError(f"No subdirectories found in {args.ref_root}")
        args.sample_obj = subdirs[0]

    sample_path = os.path.join(args.ref_root, args.sample_obj)
    indices = discover_touch_indices(sample_path, scale=args.scale)
    if not indices:
        raise FileNotFoundError(
            f"No files matching '*_scale{args.scale}_normal.jpg' found in {sample_path}"
        )

    print(f"Discovered {len(indices)} touch indices from '{sample_path}': {indices}")

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    with open(args.save_path, "w") as f:
        f.write("query\tref\n")
        for idx in indices:
            f.write(f"{idx}\t{idx}\n")

    print(f"Saved identity-mapping TSV to: {args.save_path}")


if __name__ == "__main__":
    main()
