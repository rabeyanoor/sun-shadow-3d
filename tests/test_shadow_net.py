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


def test_v2_augmentation_keeps_shapes_and_range():
    import torch

    rng = np.random.default_rng(0)
    x = torch.rand(4, 3, 64, 64)
    y = (torch.rand(4, 64, 64) > 0.5).long()
    xa = shadow_net.sun_jitter(x, rng)
    assert xa.shape == x.shape and float(xa.min()) >= 0 and float(xa.max()) <= 1
    xc, yc = shadow_net.resized_crop(x, y, rng)
    assert xc.shape == x.shape and yc.shape == y.shape and set(np.unique(yc.numpy())) <= {0, 1}


def test_concat_and_sbu_loader(tmp_path):
    from PIL import Image

    for split in ("SBU-Train", "SBU-Test"):
        for d in ("ShadowImages", "ShadowMasks"):
            (tmp_path / split / d).mkdir(parents=True)
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / split / "ShadowImages" / "a.jpg")
        Image.fromarray(np.full((8, 8), 255, np.uint8)).save(tmp_path / split / "ShadowMasks" / "a.png")
    tr, te = shadow_net.load_sbu(tmp_path, "train"), shadow_net.load_sbu(tmp_path, "test")
    assert len(tr) == len(te) == 1
    both = shadow_net.Concat(tr, te, [{"image": 1, "mask": 2}])
    assert len(both) == 3 and both[2]["mask"] == 2
    assert np.asarray(both[1]["mask"]).min() == 255 and np.asarray(both[0]["image"]).shape == (8, 8, 3)
