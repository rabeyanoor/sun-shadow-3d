"""Shadow-detector metric and preprocessing (no weights needed)."""
import numpy as np

import shadow_net


def test_ber_perfect_and_chance():
    gt = np.zeros((10, 10), bool)
    gt[:5] = True
    assert shadow_net.ber(gt, gt) == 0
    assert shadow_net.ber(~gt, gt) == 100
    assert abs(shadow_net.ber(np.ones_like(gt), gt) - 50) < 1e-9


def test_preprocessing_shapes():
    img = np.zeros((300, 500, 3), np.uint8)
    assert shadow_net._tensor(img).shape == (1, 3, shadow_net.SIZE, shadow_net.SIZE)
    m = np.zeros((300, 500), np.uint8)
    m[:, :250] = 255
    y = shadow_net._mask(m)
    assert y.shape == (1, shadow_net.SIZE, shadow_net.SIZE) and set(np.unique(y.numpy())) == {0, 1}
