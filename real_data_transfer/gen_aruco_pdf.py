"""
Generate a PDF sheet of ARuCo markers (DICT_4X4_50) for printing on A4.

Each marker is rendered at the requested physical size surrounded by a quiet
zone (white border), arranged in a grid.  Multiple pages are generated when N
exceeds what fits on one page.

Usage:
  python real_data_transfer/gen_aruco_pdf.py --n 10 --marker_size 0.05
  python real_data_transfer/gen_aruco_pdf.py --n 4 --marker_size 0.08 --start_id 2
"""

import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

# ── constants ──────────────────────────────────────────────────────────────────
A4_W_MM = 210.0
A4_H_MM = 297.0
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
MAX_MARKER_ID = 50           # DICT_4X4_50 supports IDs 0–49
RENDER_PX = 600              # internal raster resolution — does not affect print size


# ── arg parsing ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate ARuCo marker PDF (A4)")
    p.add_argument("--n", type=int, default=5,
                   help="Number of markers to generate (default: 5)")
    p.add_argument("--marker_size", type=float, default=0.05,
                   help="Printed marker side length in metres (default: 0.05 = 5 cm)")
    p.add_argument("--quiet_zone", type=float, default=0.25,
                   help="Quiet-zone width as a fraction of marker_size (default: 0.25)")
    p.add_argument("--margin_mm", type=float, default=15.0,
                   help="Page margin in mm (default: 15)")
    p.add_argument("--gap_mm", type=float, default=6.0,
                   help="Gap between marker cells in mm (default: 6)")
    p.add_argument("--start_id", type=int, default=0,
                   help="First marker ID to generate (default: 0)")
    p.add_argument("--log_dir", default="log",
                   help="Output directory (default: log/)")
    return p.parse_args()


# ── pdf generation ─────────────────────────────────────────────────────────────

def generate_pdf(n: int, marker_size_m: float, quiet_zone_frac: float,
                 margin_mm: float, gap_mm: float,
                 start_id: int, log_dir: str) -> str:
    """Lay out N markers on A4 pages and save as PDF.  Returns the output path."""

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)

    size_mm = marker_size_m * 1_000.0          # metres → mm
    qz_mm = size_mm * quiet_zone_frac          # quiet-zone width on each side
    cell_mm = size_mm + 2.0 * qz_mm            # cell = marker + quiet zone both sides

    usable_w = A4_W_MM - 2.0 * margin_mm
    usable_h = A4_H_MM - 2.0 * margin_mm

    cols = max(1, int((usable_w + gap_mm) / (cell_mm + gap_mm)))
    rows = max(1, int((usable_h + gap_mm) / (cell_mm + gap_mm)))
    per_page = cols * rows
    n_pages = (n + per_page - 1) // per_page

    os.makedirs(log_dir, exist_ok=True)
    end_id = start_id + n - 1
    output_path = os.path.join(log_dir, f"aruco_{start_id:02d}_to_{end_id:02d}.pdf")

    fig_w = A4_W_MM / 25.4   # inches
    fig_h = A4_H_MM / 25.4

    with PdfPages(output_path) as pdf:
        for page in range(n_pages):
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

            # Coordinate system: origin top-left, y increases downward (matches A4 layout)
            ax.set_xlim(0, A4_W_MM)
            ax.set_ylim(A4_H_MM, 0)
            ax.set_aspect("equal")
            ax.axis("off")

            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            for slot in range(per_page):
                global_idx = page * per_page + slot
                if global_idx >= n:
                    break
                marker_id = start_id + global_idx

                col = slot % cols
                row = slot // cols

                # Top-left corner of this cell
                cx = margin_mm + col * (cell_mm + gap_mm)
                cy = margin_mm + row * (cell_mm + gap_mm)

                # White quiet-zone background with cut line
                ax.add_patch(mpatches.Rectangle(
                    (cx, cy), cell_mm, cell_mm,
                    facecolor="white", edgecolor="black", linewidth=0.5, zorder=1,
                ))

                # Marker image
                img = cv2.aruco.generateImageMarker(dictionary, marker_id, RENDER_PX)
                mx, my = cx + qz_mm, cy + qz_mm
                # extent = [left, right, bottom, top] in data coords (y-down: bottom > top)
                ax.imshow(
                    img, cmap="gray", vmin=0, vmax=255,
                    extent=[mx, mx + size_mm, my + size_mm, my],
                    origin="upper", aspect="auto", zorder=2,
                    interpolation="nearest",
                )

                # Label centred below the cell
                label_y = cy + cell_mm + 2.5
                ax.text(
                    cx + cell_mm / 2, label_y,
                    f"ID {marker_id}   {size_mm:.0f} mm",
                    ha="center", va="top",
                    fontsize=max(5, min(8, size_mm * 0.14)),
                    color="black", zorder=3,
                )

            # Page footer
            ax.text(
                A4_W_MM / 2, A4_H_MM - 4,
                f"ARuCo DICT_4X4_50  ·  marker size {size_mm:.0f} mm  "
                f"·  page {page + 1}/{n_pages}",
                ha="center", va="bottom", fontsize=7, color="#666666",
            )

            pdf.savefig(fig, dpi=300, bbox_inches=None)
            plt.close(fig)

    print(
        f"Saved → {output_path}\n"
        f"  markers : ID {start_id}–{end_id}  ({n} total)\n"
        f"  layout  : {cols} col × {rows} row = {per_page}/page  ({n_pages} page(s))\n"
        f"  cell    : {cell_mm:.1f} mm  (marker {size_mm:.0f} mm + "
        f"quiet zone {qz_mm:.1f} mm each side)"
    )
    return output_path


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    end_id = args.start_id + args.n - 1
    if args.start_id < 0 or end_id >= MAX_MARKER_ID:
        raise SystemExit(
            f"DICT_4X4_50 only supports IDs 0–{MAX_MARKER_ID - 1}. "
            f"Requested {args.start_id}–{end_id}."
        )

    generate_pdf(
        n=args.n,
        marker_size_m=args.marker_size,
        quiet_zone_frac=args.quiet_zone,
        margin_mm=args.margin_mm,
        gap_mm=args.gap_mm,
        start_id=args.start_id,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
