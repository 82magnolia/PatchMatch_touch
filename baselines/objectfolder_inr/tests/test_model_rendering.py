import numpy as np
import torch

from objectfolder_inr.model import PositionalEncoder, TouchNet
from objectfolder_inr.rendering import Renderer, height_to_normal_bgr


def test_positional_encoding_and_network_shape():
    encoder = PositionalEncoder(9, 3)
    assert encoder(torch.zeros(4, 9)).shape == (4, 63)
    model = TouchNet(levels=2, depth=4, width=16)
    result = model(torch.zeros(5, 9))
    assert result.shape == (5, 1)
    assert torch.all((result >= 0.0) & (result <= 1.0))


def test_height_normal_render_shape():
    yy, xx = np.mgrid[:12, :16]
    image = height_to_normal_bgr((xx + yy).astype(np.float32))
    assert image.shape == (12, 16, 3)
    assert image.dtype == np.uint8


def test_zero_depth_is_flat_no_contact_normal(tmp_path):
    renderer = Renderer("tactile_normal", None, tmp_path)
    image = renderer.render(np.random.rand(12, 16), 0.0, (32, 24))
    assert image.shape == (24, 32, 3)
    assert np.all(image == np.array([255, 128, 128], dtype=np.uint8))
