"""Small helpers shared by the teaser and method-overview figure scripts.

Everything is laid out in inches from the top-left corner of the page, which
keeps a one-column figure predictable: the numbers in the layout code are the
numbers that end up on paper.
"""
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch   # noqa: E402

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
ASSETS = f"{ROOT}/log/paper_job04_paper_figures/assets"

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,      # editable text in the PDF, as venues expect
    "ps.fonttype": 42,
})

# One accent colour for "reference / given" material and one for "query /
# predicted", used consistently across both figures.
C_REF = "#1f6fb4"
C_QRY = "#c1452a"
C_GT = "#3a7d3a"
C_BOX = "#d21f1f"        # the 1x sensor footprint outline
C_LINE = "#444444"


def load(path):
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def white_bg(img, thr=96):
    """Repaint a normal render's empty background from black to white.

    Taxim renders empty space as exactly (0, 0, 0), while surface pixels encode a
    unit normal and so are never dark: across the benchmark renders the darkest
    surface pixel has a maximum channel of about 122, and no pixel at all falls
    between 64 and 120. A single threshold on the brightest channel therefore
    separates the two cleanly, JPEG ringing included.
    """
    out = img.copy()
    out[img.max(axis=2) < thr] = 255
    return out


def load_normal(path):
    """Load a surface-normal render for display: background repainted white.

    Use this rather than plain load() for anything that came out of Taxim's
    normal renderer, so every figure shows the geometry on white instead of on
    a black frame. Feature matching and retrieval still run on the untouched
    renders -- only what the reader sees is repainted.
    """
    return white_bg(load(path))


def sensor_box(img, scale=25):
    """Outline the 1x sensor footprint inside a wider-field-of-view render.

    Taxim keeps the rendered window at a fixed pixel size and shrinks the
    object instead, so the area a 1x render covers appears in a `scale` render
    as a centred box of linear size scale/100 (a quarter of the frame for the
    4x renders used for matching).
    """
    ratio = 100.0 / float(scale)
    out = img.copy()
    h, w = out.shape[:2]
    bh, bw = int(round(h / ratio)), int(round(w / ratio))
    y0, x0 = (h - bh) // 2, (w - bw) // 2
    cv2.rectangle(out, (x0, y0), (x0 + bw - 1, y0 + bh - 1),
                  tuple(int(c * 255) for c in matplotlib.colors.to_rgb(C_BOX)),
                  max(2, h // 120))
    return out


class Page:
    """A figure whose contents are positioned in inches from the top-left."""

    def __init__(self, width, height, dpi=400, facecolor="white"):
        self.w, self.h = width, height
        self.fig = plt.figure(figsize=(width, height), dpi=dpi, facecolor=facecolor)

    def _rect(self, x, y, w, h):
        return [x / self.w, 1.0 - (y + h) / self.h, w / self.w, h / self.h]

    def ax(self, x, y, w, h, z=1):
        a = self.fig.add_axes(self._rect(x, y, w, h), zorder=z)
        a.set_xticks([])
        a.set_yticks([])
        return a

    def img(self, x, y, w, h, im, edge=C_LINE, lw=0.6, z=2):
        a = self.ax(x, y, w, h, z=z)
        a.imshow(im, interpolation="lanczos")
        for s in a.spines.values():
            s.set_linewidth(lw)
            s.set_color(edge)
            s.set_visible(lw > 0)
        return a

    def text(self, x, y, s, size=6.0, ha="left", va="center", color="black",
             weight="normal", style="normal", z=5, **kw):
        return self.fig.text(x / self.w, 1.0 - y / self.h, s, fontsize=size, ha=ha,
                             va=va, color=color, fontweight=weight, fontstyle=style,
                             zorder=z, **kw)

    def panel(self, x, y, w, h, facecolor="#f4f5f7", edgecolor="#d5d8dd", lw=0.6,
              radius=0.02, z=0):
        a = self.fig.add_axes([0, 0, 1, 1], zorder=z)
        a.set_axis_off()
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.patch.set_alpha(0)
        r = self._rect(x, y, w, h)
        a.add_patch(FancyBboxPatch((r[0], r[1]), r[2], r[3],
                                   boxstyle=f"round,pad=0,rounding_size={radius}",
                                   facecolor=facecolor, edgecolor=edgecolor,
                                   linewidth=lw, transform=a.transAxes))
        return a

    def _overlay(self, z=4):
        a = self.fig.add_axes([0, 0, 1, 1], zorder=z)
        a.set_axis_off()
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.patch.set_alpha(0)
        return a

    def arrow(self, x0, y0, x1, y1, color=C_LINE, lw=0.9, mut=4.0, style="-|>",
              connection="arc3,rad=0", z=4):
        a = self._overlay(z)
        p0 = (x0 / self.w, 1 - y0 / self.h)
        p1 = (x1 / self.w, 1 - y1 / self.h)
        a.add_patch(FancyArrowPatch(p0, p1, transform=a.transAxes,
                                    arrowstyle=style, mutation_scale=mut,
                                    linewidth=lw, color=color,
                                    connectionstyle=connection,
                                    shrinkA=0, shrinkB=0))
        return a

    def line(self, pts, color=C_LINE, lw=0.7, ls="-", z=4, alpha=1.0):
        a = self._overlay(z)
        xs = [p[0] / self.w for p in pts]
        ys = [1 - p[1] / self.h for p in pts]
        a.plot(xs, ys, color=color, lw=lw, ls=ls, alpha=alpha,
               transform=a.transAxes, solid_capstyle="round")
        return a

    def box(self, x, y, w, h, facecolor="white", edgecolor=C_LINE, lw=0.7,
            radius=0.012, z=3, dashed=False):
        a = self._overlay(z)
        r = self._rect(x, y, w, h)
        a.add_patch(FancyBboxPatch((r[0], r[1]), r[2], r[3],
                                   boxstyle=f"round,pad=0,rounding_size={radius}",
                                   facecolor=facecolor, edgecolor=edgecolor,
                                   linewidth=lw, linestyle="--" if dashed else "-",
                                   transform=a.transAxes))
        return a

    def save(self, stem):
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        self.fig.savefig(f"{stem}.png", facecolor=self.fig.get_facecolor())
        self.fig.savefig(f"{stem}.pdf", facecolor=self.fig.get_facecolor())
        plt.close(self.fig)
        print("wrote", f"{stem}.png", "and .pdf")
