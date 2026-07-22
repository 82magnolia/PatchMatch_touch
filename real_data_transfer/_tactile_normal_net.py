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


def boundary_band(mask, band_px):
    """Boolean mask of pixels within band_px of the contact-mask boundary, on
    both sides -- the only region the Poisson blend is allowed to modify."""
    m = mask.astype(np.uint8)
    din = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    return (mask & (din <= band_px)) | (~mask & (dout <= band_px))


def poisson_blend_normals(nx, ny, mask, band_px, use_guidance=True):
    """Poisson/gradient-domain blend of the (nx, ny) seam, both channels at once.

    Only pixels within band_px of the contact boundary are solved; the deep
    interior (net values) and deep exterior (0) are Dirichlet-fixed, so interior
    detail is left untouched while the seam becomes a smooth ramp to background.
    With use_guidance the band still imports the net's in-mask gradients so rim
    detail carries through. The sparse system depends only on the band geometry,
    so it is factored once and applied to both channels. Returns (nx, ny).
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import factorized

    nx = np.asarray(nx, dtype=np.float32)
    ny = np.asarray(ny, dtype=np.float32)
    band = boundary_band(mask, band_px)
    ys, xs = np.where(band)
    n = len(ys)
    if n == 0:
        return nx.copy(), ny.copy()

    h, w = nx.shape
    idx = -np.ones((h, w), dtype=np.int64)
    idx[ys, xs] = np.arange(n)
    rows, cols, data = [], [], []
    bx = np.zeros(n, dtype=np.float64)
    by = np.zeros(n, dtype=np.float64)
    for i in range(n):
        y, x = int(ys[i]), int(xs[i])
        diag = 0.0
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = y + dy, x + dx
            if yy < 0 or yy >= h or xx < 0 or xx >= w:
                continue  # image border: Neumann (drop the neighbour)
            diag += 1.0
            j = idx[yy, xx]
            if j >= 0:
                rows.append(i); cols.append(j); data.append(-1.0)
            elif mask[yy, xx]:  # fixed neighbour: net value inside, else 0
                bx[i] += nx[yy, xx]
                by[i] += ny[yy, xx]
            if use_guidance and mask[y, x] and mask[yy, xx]:
                bx[i] += nx[y, x] - nx[yy, xx]
                by[i] += ny[y, x] - ny[yy, xx]
        rows.append(i); cols.append(i); data.append(diag)

    solve = factorized(csr_matrix((data, (rows, cols)), shape=(n, n)).tocsc())
    onx, ony = nx.copy(), ny.copy()
    onx[ys, xs] = solve(bx).astype(np.float32)
    ony[ys, xs] = solve(by).astype(np.float32)
    return onx, ony


def frame_to_normals(frame_bgr, net, device, contact_mask=None,
                     marker_range=(0, 70), poisson_band_px=0):
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

    poisson_band_px: if > 0, Poisson-blend the (nx, ny) field within this many
    pixels of the mask boundary (see poisson_blend_normals) to soften the hard
    foreground/background seam while leaving the interior untouched. 0 = hard
    boundary.
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

    rgb_norm = frame_rgb[contact_mask] / 255.0
    px = np.vstack(np.where(contact_mask)).T.astype(np.float64)
    px[:, 0] /= h
    px[:, 1] /= w

    features = np.column_stack((rgb_norm, px))
    features_t = torch.from_numpy(features).float().to(device)

    with torch.no_grad():
        out = net(features_t).cpu().numpy()

    normal_x = np.zeros((h, w), dtype=np.float32)
    normal_y = np.zeros((h, w), dtype=np.float32)
    normal_x[contact_mask] = out[:, 0]
    normal_y[contact_mask] = out[:, 1]

    if poisson_band_px and poisson_band_px > 0:
        normal_x, normal_y = poisson_blend_normals(
            normal_x, normal_y, contact_mask, poisson_band_px)

    # unit_normals enforces nx^2 + ny^2 + nz^2 == 1 for every pixel (the raw
    # network (nx, ny) and the Poisson solve can both leave the unit disk).
    return unit_normals(normal_x, normal_y)
