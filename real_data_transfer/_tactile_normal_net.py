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


def frame_to_normals(frame_bgr, net, device, marker_range=(0, 70)):
    """GelSight BGR frame -> (H,W,3) float32 unit normal map (nx,ny,nz).

    marker_range: grayscale intensity range treated as ARuCO/marker dots,
    which are excluded from the network input and held at (0,0,1)
    (matches Reconstruction3D.get_depthmap's markers_threshold masking).
    Pass None to run the network on every pixel.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]

    contact_mask = np.ones((h, w), dtype=bool)
    if marker_range is not None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        marker_mask = cv2.inRange(gray, marker_range[0], marker_range[1]) > 0
        contact_mask = ~marker_mask

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

    normal_z = np.sqrt(np.clip(1.0 - normal_x ** 2 - normal_y ** 2, 0.0, 1.0))
    return np.stack([normal_x, normal_y, normal_z], axis=-1)
