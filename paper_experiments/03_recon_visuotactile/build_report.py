"""HTML + Markdown report for the 3D reconstruction / visuo-tactile simulation job."""
import base64
import glob
import os

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
OUT = f"{ROOT}/log/paper_job03_recon_visuotactile"
HERE = os.path.dirname(os.path.abspath(__file__))

PRIMARY = "951_5"
ALTERNATES = ["977_1", "981_3", "969_5", "967_1", "965_3"]


def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode("ascii")


def fig(key):
    p = f"{OUT}/figure_{key}.png"
    if not os.path.exists(p):
        return f"<p><i>figure_{key}.png not generated</i></p>"
    return f'<img style="width:100%;display:block;border:1px solid #ddd" src="data:image/png;base64,{b64(p)}">'


n_assets = len(glob.glob(f"{OUT}/assets/*/*.png"))
has_video = os.path.exists(f"{OUT}/video_{PRIMARY}.mp4")

alt_html = "".join(
    f"<figure>{fig(k)}<figcaption>Alternative example &mdash; object {k.split('_')[0]}, "
    f"touch {k.split('_')[1]}.</figcaption></figure>" for k in ALTERNATES
    if os.path.exists(f"{OUT}/figure_{k}.png"))

HTML = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Job 3 &mdash; 3D reconstruction and visuo-tactile simulation</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
      max-width:1150px;margin:2rem auto;padding:0 1.2rem;line-height:1.6;color:#1b1b1f}}
 h1{{border-bottom:2px solid #333;padding-bottom:.4rem}}
 h2{{margin-top:2.4rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}}
 table{{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.93rem}}
 th,td{{border:1px solid #ddd;padding:.42rem .7rem;text-align:left;vertical-align:top}}
 th{{background:#f7f7f9;width:210px}}
 code{{background:#f4f4f6;padding:.1rem .3rem;border-radius:3px;font-size:.88em}}
 figure{{margin:1.4rem 0}} figcaption{{color:#555;font-size:.9rem;margin-top:.4rem}}
 .note{{background:#eef6ff;border-left:4px solid #3d7fd1;padding:.7rem 1rem;margin:1rem 0}}
 ol li{{margin:.3rem 0}}
</style></head><body>

<h1>Job 3 &mdash; 3D surface reconstruction and visuo-tactile sensor simulation</h1>

<p>This job shows what the predicted tactile videos are <i>good for</i>. Our method outputs a
tactile normal map for every frame &mdash; a picture of which way the gel surface is tilted at
each pixel. Two useful things follow from that, without ever touching the object:</p>
<ol>
 <li><b>3D shape.</b> Tilts can be integrated into heights, so each predicted frame becomes a
     small 3D surface patch of the object at the touched spot.</li>
 <li><b>A simulated camera-based tactile sensor.</b> Feeding those heights through Taxim's
     optical model produces the RGB image a real GelSight-style sensor would have shown.</li>
</ol>

<h2>The pipeline, step by step</h2>
<table>
<tr><th>1. Reference touch</th><td>An existing tactile video recorded somewhere else on the object.</td></tr>
<tr><th>2. Our prediction</th><td>Coarse alignment by feature matching, then the refinement network
    (<code>log/rebot_checkpoints_S_geomcat_film/best.pth</code>), giving a predicted tactile
    normal video at the new, never-touched query location.</td></tr>
<tr><th>3. Normals &rarr; heights</th><td>Each predicted normal map is converted to slopes and integrated
    into a heightmap with a Poisson solver (<code>poisson_dct_neumann</code>). The overall tilt of the
    result is fitted and removed, since integration only recovers shape up to a plane.</td></tr>
<tr><th>4. Heights &rarr; 3D relief</th><td>The heightmap is drawn as a shaded 3D surface. The shading is a plain matte (Lambertian) response to a single light, computed from the surface's own slopes &mdash; the same style used in <code>train_refine_scripts/time_cond_sweep/height3d_geomcat_film.py</code>. It is used in preference to matplotlib's built-in hillshade because that one produces contour-like rings on these video-compressed normals.</td></tr>
<tr><th>5. Heights &rarr; RGB</th><td>Taxim's calibrated optical model (a gradient-to-colour lookup table
    plus the gel background image, <code>Taxim/calibs/polycalib.npz</code> and <code>dataPack.npz</code>)
    turns the heightmap into the RGB image a real sensor would produce, at the sensor's native
    480 &times; 640 resolution.</td></tr>
</table>

<div class="note">Steps 3&ndash;5 involve no learning and no extra ground truth: they are
deterministic transforms of our network's own output. So everything below is obtainable at
deployment time from a single reference touch.</div>

<h2>Main figure</h2>
<figure>{fig(PRIMARY)}
<figcaption>Object {PRIMARY.split('_')[0]}, touch {PRIMARY.split('_')[1]}. Each column is one frame
of the press, moving left to right through the contact. Row 1: the reference tactile normal video
we transfer from. Row 2: our predicted tactile normal video at the query pose. Row 3: the shaded 3D
surface obtained by integrating row 2 into a heightmap. Row 4: the RGB visuo-tactile frames obtained
by running row 3 through Taxim's optical simulation. Note that the reference contact is roughly
horizontal while the query contact is tilted &mdash; the prediction follows the query geometry, not
the reference.</figcaption></figure>

<h2>Alternative examples</h2>
<p>Generated so a different example can be swapped into the paper without re-running anything.</p>
{alt_html}

{'<h2>Video</h2><p>A full-length 50-frame video of the main example (reference | predicted normal | 3D relief | simulated RGB) is at <code>log/paper_job03_recon_visuotactile/video_' + PRIMARY + '.mp4</code>.</p>' if has_video else ''}

<h2>Choices worth knowing about</h2>
<ul>
<li><b>Which frames become columns.</b> The first and last frames of a press are no-contact
    readings of flat gel. Integrating those amplifies video-compression noise into meaningless
    ripples, so the columns are spread evenly over the part of the press where contact is
    actually present.</li>
<li><b>Which way is up.</b> Integrating slopes cannot tell a bump from a dent, so the sign is
    chosen such that the contact region sits above the surrounding background.</li>
<li><b>Smoothing.</b> The heightmap is lightly blurred before being shaded, purely so the relief
    reads cleanly; the RGB simulation uses the unsmoothed heightmap.</li>
</ul>

<h2>Where the assets are</h2>
<ul>
<li><code>log/paper_job03_recon_visuotactile/figure_&lt;object&gt;_&lt;touch&gt;.png</code> &mdash; stitched figures</li>
<li><code>log/paper_job03_recon_visuotactile/assets/&lt;object&gt;_&lt;touch&gt;/col&lt;NN&gt;_f&lt;FFF&gt;_row&lt;R&gt;_*.png</code>
    &mdash; every cell as a separate full-resolution PNG ({n_assets} files), so the figure can be
    re-laid-out in LaTeX</li>
<li><code>paper_experiments/03_recon_visuotactile/make_recon_figure.py</code> &mdash; the generator</li>
</ul>
</body></html>
"""

MD = f"""# Job 3 — 3D surface reconstruction and visuo-tactile sensor simulation

## What the figure shows

Four rows, columns = frames of one touch:

1. reference tactile normal video (the example transferred from)
2. predicted tactile normal video (our refinement network)
3. shaded 3D relief from the predicted heightmap
4. simulated RGB visuo-tactile frames (Taxim optical model)

Primary example: object {PRIMARY.split('_')[0]}, touch {PRIMARY.split('_')[1]}
(`log/paper_job03_recon_visuotactile/figure_{PRIMARY}.png`).
Alternates generated: {', '.join(ALTERNATES)}.

## Method details for the paper text

- Normals -> heights: invert Taxim's normal encoding to slopes
  (`gx = -nx/nz`, `gy = -ny/nz`), integrate with a discrete-cosine-transform Poisson
  solver with Neumann boundary conditions (`poisson_dct_neumann`), then subtract a
  least-squares plane fit (integration is only defined up to a plane).
- Sign disambiguation: the heightmap is flipped, if needed, so the contact region is
  raised relative to the background.
- Visualization: the heightmap is drawn as a shaded 3D surface with plain matte
  (Lambertian) shading computed from its own slopes, light direction (-0.5, -0.6, 0.7),
  ambient 0.28 + diffuse 0.72, viewed at elevation 55 / azimuth -62. This is the same
  style as `train_refine_scripts/time_cond_sweep/height3d_geomcat_film.py`; matplotlib's
  built-in hillshade was avoided because it produces contour-like rings on these
  video-compressed normals.
- Heights -> RGB: Taxim optical simulation at the sensor's native 480 x 640 resolution,
  using the calibrated gradient-to-RGB polynomial lookup table (`Taxim/calibs/polycalib.npz`)
  plus the smoothed gel background frame (`dataPack.npz`).
- Every step after the network is deterministic and needs no ground truth, so the whole
  chain runs at deployment time from one reference touch.

## Sentence-ready summary

> Because our method predicts tactile surface normals, its output can be integrated into a
> heightmap with a Poisson solver, yielding a 3D reconstruction of the object surface at a
> location that was never physically touched. Feeding the same heightmap through Taxim's
> calibrated optical model further produces the RGB image a camera-based tactile sensor
> would have measured, i.e. a virtual visuo-tactile measurement. Both steps are deterministic
> post-processing of the predicted normals and require no additional supervision.

## Caveat

The no-contact frames at the start and end of a press integrate into meaningless ripples
(there is no signal to integrate); figure columns are therefore sampled from the
in-contact portion of the press.
"""

open(os.path.join(HERE, "report.html"), "w").write(HTML)
open(os.path.join(HERE, "results.md"), "w").write(MD)
open(f"{OUT}/report.html", "w").write(HTML)
print("wrote report.html + results.md")
