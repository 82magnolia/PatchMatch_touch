"""
Standalone RGB2NormNet inference for tactile (GelSight) normal maps.

Self-contained port of gsrobotics' utilities.reconstruction.RGB2NormNet /
Reconstruction3D.get_depthmap, trimmed to just the per-pixel normal
prediction -- no Poisson depth integration, no depth-zeroing state --
since process_single_shot.py only needs a color-coded normal video,
not a depth map.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class RGB2NormNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 2)
        self.drop_layer = nn.Dropout(p=0.05)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc2(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc3(x))
        x = self.drop_layer(x)
        return self.fc4(x)


def load_normal_net(net_path, device):
    """Load an RGB2NormNet checkpoint (e.g. gsnormal_models/nnmini.pt)."""
    net = RGB2NormNet().float().to(device)
    state = torch.load(net_path, map_location=device)
    net.load_state_dict(state["state_dict"])
    net.eval()
    return net


def unit_normals(nx, ny):
    """(nx, ny) tangential components -> (H,W,3) float32 unit normal map.

    Projects any pixel whose (nx, ny) falls outside the unit disk back onto it,
    then recovers nz = sqrt(1 - nx^2 - ny^2), so nx^2 + ny^2 + nz^2 == 1 holds
    for every pixel.
    """
    nx = np.asarray(nx, dtype=np.float32).copy()
    ny = np.asarray(ny, dtype=np.float32).copy()
    r2 = nx ** 2 + ny ** 2
    over = r2 > 1.0
    if over.any():
        inv = 1.0 / np.sqrt(r2[over])
        nx[over] *= inv
        ny[over] *= inv
    nz = np.sqrt(np.clip(1.0 - nx ** 2 - ny ** 2, 0.0, 1.0))
    return np.stack([nx, ny, nz], axis=-1)


def poisson_blend_normals(nx, ny, mask, *, use_guidance=True):
    """Poisson/gradient-domain blend of the seam, both channels at once.

    Every pixel of the contact mask is an unknown; the whole background is
    Dirichlet-fixed at 0 (flat). With use_guidance the right-hand side is the
    input field's own Laplacian, so the solution is that field plus the harmonic
    correction bringing it to 0 at the mask border: every interior gradient is
    preserved exactly, while any low-frequency offset between foreground and
    background is absorbed across the whole region. Standard Poisson editing
    (Perez et al.).

    Only in-mask neighbour pairs contribute guidance; pairs straddling the mask
    boundary are left out on purpose, so the solution is free to meet the flat
    background smoothly rather than reproducing the step there. Pixels at the
    image edge simply drop the missing neighbour, which is a Neumann (zero-flux)
    condition -- masks that run to the frame border float rather than being
    pulled to 0.

    The sparse system depends only on the mask geometry, so it is factored once
    and applied to both channels. Returns (nx, ny).

    use_guidance is keyword-only on purpose: this signature previously took a
    band_px in that slot, and a leftover positional call must fail loudly rather
    than quietly reinterpret a band width as a truthy use_guidance.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import factorized

    nx = np.asarray(nx, dtype=np.float32)
    ny = np.asarray(ny, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.where(mask)
    n = len(ys)
    if n == 0:
        return nx.copy(), ny.copy()

    h, w = nx.shape
    idx = np.full((h, w), -1, dtype=np.int64)
    idx[ys, xs] = np.arange(n)
    here = np.arange(n)

    rows, cols = [here], [here]
    diag = np.zeros(n, dtype=np.float64)
    data = [diag]  # accumulated in the loop below; held here by reference
    bx = np.zeros(n, dtype=np.float64)
    by = np.zeros(n, dtype=np.float64)

    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        yy, xx = ys + dy, xs + dx
        inb = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)  # else Neumann: drop
        diag[inb] += 1.0

        j = np.full(n, -1, dtype=np.int64)
        j[inb] = idx[yy[inb], xx[inb]]
        # In-mask neighbours are unknowns; in-bounds ones outside the mask are
        # the Dirichlet boundary, and since that value is 0 they contribute
        # nothing beyond the diagonal count already added above.
        unknown = j >= 0
        rows.append(here[unknown])
        cols.append(j[unknown])
        data.append(np.full(int(unknown.sum()), -1.0))

        if use_guidance:
            u = unknown
            bx[u] += nx[ys[u], xs[u]] - nx[yy[u], xx[u]]
            by[u] += ny[ys[u], xs[u]] - ny[yy[u], xx[u]]

    A = csr_matrix((np.concatenate(data),
                    (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    solve = factorized(A.tocsc())
    onx, ony = nx.copy(), ny.copy()
    onx[ys, xs] = solve(bx).astype(np.float32)
    ony[ys, xs] = solve(by).astype(np.float32)
    return onx, ony


def net_nxny(frame_bgr, net, device, pixels=None):
    """Run RGB2NormNet on `pixels` (default: every pixel) -> (nx, ny) planes.

    Pixels not in `pixels` are left at 0. The network input is (r, g, b, y/H,
    x/W) exactly as in gsrobotics' Reconstruction3D.get_depthmap.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    if pixels is None:
        pixels = np.ones((h, w), dtype=bool)

    normal_x = np.zeros((h, w), dtype=np.float32)
    normal_y = np.zeros((h, w), dtype=np.float32)
    if not pixels.any():
        return normal_x, normal_y

    rgb_norm = frame_rgb[pixels] / 255.0
    px = np.vstack(np.where(pixels)).T.astype(np.float64)
    px[:, 0] /= h
    px[:, 1] /= w
    features_t = torch.from_numpy(
        np.column_stack((rgb_norm, px))).float().to(device)
    with torch.no_grad():
        out = net(features_t).cpu().numpy()

    normal_x[pixels] = out[:, 0]
    normal_y[pixels] = out[:, 1]
    return normal_x, normal_y


# The network's 2 outputs are unconstrained, so (nx, ny) can land outside the
# unit disk (~0.3% of pixels, up to 3% on some frames), where nz would be the
# square root of a negative number. Even just inside the rim, nz -> 0 sends the
# slope -nx/nz to infinity: unclamped, real frames produce |slope| ~ 1300, i.e.
# 89.96 degree "walls" the gel cannot physically take. Clamping the tangential
# radius to sin(MAX_SLOPE_DEG) handles both cases with one operation, and keeps
# nz >= cos(MAX_SLOPE_DEG) so the division is always well conditioned.
# 85 degrees sits far above the real signal -- the slope magnitude's 99.9th
# percentile over in-disk pixels is 3.6 (74.6 degrees) -- so this clips only
# 0.008% of valid pixels while removing every blow-up.
MAX_SLOPE_DEG = 85.0


def nxny_to_gradients(nx, ny):
    """Unit-normal tangential components -> surface slopes (dz/dx, dz/dy).

    Same formula as Reconstruction3D.get_depthmap (gx = -nx/nz, gy = -ny/nz),
    but with the tangential radius clamped first (see MAX_SLOPE_DEG). gsrobotics
    instead substitutes the frame's mean nz wherever the square root goes NaN;
    radial clamping is preferred here because it is local, leaves every in-range
    pixel untouched, and matches how unit_normals already projects strays back
    onto the disk.
    """
    nx = np.asarray(nx, dtype=np.float32)
    ny = np.asarray(ny, dtype=np.float32)
    r_max = np.float32(np.sin(np.radians(MAX_SLOPE_DEG)))
    r = np.hypot(nx, ny)
    scale = np.minimum(np.float32(1.0), r_max / np.maximum(r, np.float32(1e-12)))
    nx, ny = nx * scale, ny * scale
    nz = np.sqrt(np.clip(1.0 - nx ** 2 - ny ** 2, 0.0, 1.0), dtype=np.float32)
    return (-nx / nz).astype(np.float32), (-ny / nz).astype(np.float32)


def gradients_to_normals(gx, gy):
    """Surface slopes -> (H,W,3) unit normal map, the inverse of the above.

    Identical construction to Taxim's height_map_to_normals
    (normal = [-dzdx, -dzdy, 1] / norm), so both pipelines end up expressing
    the same quantity the same way. Unit length holds by construction, so no
    projection back onto the unit disk is needed.
    """
    denom = np.sqrt(gx ** 2 + gy ** 2 + 1.0)
    return np.stack([-gx / denom, -gy / denom, 1.0 / denom],
                    axis=-1).astype(np.float32)


def baseline_gradients(frame_bgr, net, device):
    """Surface slopes the network reports for a *no-contact* reference frame.

    RGB2NormNet is calibrated per sensor, and on this GelSight an undeformed
    gel does NOT map to (0, 0, 1): the raw prediction sits at roughly
    (nx, ny) = (+0.17, -0.53) (nz ~ 0.76, i.e. a phantom ~41 degree tilt),
    with a further fixed spatial pattern (per-pixel std ~0.13 / ~0.25) from
    the LED illumination gradient. Passing this field back into
    frame_to_normals as `baseline` subtracts both, so a flat gel reads as
    exactly (0, 0, 1) -- the same zero Taxim's height-map normals use.

    Subtraction happens in *gradient* space, not on (nx, ny), because that is
    what gsrobotics actually does: Reconstruction3D.get_depthmap averages the
    first 50 untouched frames into `depth_map_zero` and subtracts it from
    depth, and depth is poisson_dct_neumann(gx, gy) -- a linear operator on
    the gradients. Subtracting a constant depth field is therefore exactly
    subtracting the corresponding constant gradient field. Slopes are also the
    physically meaningful thing to difference (a fixed tilt of the reference
    surface); differencing unit-vector components is not a well-defined
    geometric operation and can leave the unit disk.
    """
    return nxny_to_gradients(*net_nxny(frame_bgr, net, device))


def frame_to_normals(frame_bgr, net, device, contact_mask=None,
                     marker_range=(0, 70), poisson_blend=False, baseline=None):
    """GelSight BGR frame -> (H,W,3) float32 unit normal map (nx,ny,nz).

    contact_mask: optional (H,W) boolean mask of pixels to run the network on.
    When given it fully determines the gating (e.g. compute_contact_mask's
    image-diff contact footprint) and marker_range is ignored. Pixels outside
    the mask are held at (0,0,1).

    marker_range: fallback gating when contact_mask is None -- grayscale
    intensity range treated as ARuCO/marker dots, which are excluded from the
    network input and held at (0,0,1) (matches Reconstruction3D.get_depthmap's
    markers_threshold masking). Pass None (with no contact_mask) to run the
    network on every pixel.

    poisson_blend: True to Poisson-blend the field over the whole foreground
    against the flat background (see poisson_blend_normals), False for a hard
    mask boundary.

    baseline: optional (gx, gy) tuple from baseline_gradients() for this
    session's no-contact frame. When given, the network is run on every pixel,
    converted to surface slopes, and the baseline slopes are subtracted, so the
    output measures deformation *relative to the flat gel* rather than the
    network's absolute (biased) prediction. Strongly recommended -- without it
    the in-mask region carries a large DC offset that the (0,0,1) background
    does not. With a baseline the masking, the Poisson blend and the final
    normal conversion all happen in gradient space.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]

    if contact_mask is not None:
        contact_mask = np.asarray(contact_mask, dtype=bool)
    elif marker_range is not None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        marker_mask = cv2.inRange(gray, marker_range[0], marker_range[1]) > 0
        contact_mask = ~marker_mask
    else:
        contact_mask = np.ones((h, w), dtype=bool)

    # Empty mask (e.g. a no-contact base frame) -> flat (0,0,1) everywhere;
    # feeding the network an empty batch would error.
    if not contact_mask.any():
        flat = np.zeros((h, w, 3), dtype=np.float32)
        flat[..., 2] = 1.0
        return flat

    if baseline is not None:
        # Run everywhere so the baseline can be subtracted per pixel, in the
        # gradient space gsrobotics zeroes in (see baseline_gradients), then
        # gate: outside the contact mask the deformation is zero by definition,
        # and zero slope is exactly the flat (0,0,1) background.
        grad_x, grad_y = nxny_to_gradients(*net_nxny(frame_bgr, net, device))
        grad_x = grad_x - baseline[0]
        grad_y = grad_y - baseline[1]
        grad_x[~contact_mask] = 0.0
        grad_y[~contact_mask] = 0.0
        if poisson_blend:
            grad_x, grad_y = poisson_blend_normals(
                grad_x, grad_y, contact_mask)
        return gradients_to_normals(grad_x, grad_y)

    normal_x, normal_y = net_nxny(frame_bgr, net, device, contact_mask)

    if poisson_blend:
        normal_x, normal_y = poisson_blend_normals(
            normal_x, normal_y, contact_mask)

    # unit_normals enforces nx^2 + ny^2 + nz^2 == 1 for every pixel (the raw
    # network (nx, ny) and the Poisson solve can both leave the unit disk).
    return unit_normals(normal_x, normal_y)
