"""A learned shadow detector: SegFormer fine-tuned on ISTD (shadow image + shadow mask).

The pilot used a dark-asphalt threshold, and the synthetic study (notebook 04) showed that
this is where the shape/focal signal got lost. Hugging Face has no ready shadow detector,
so we train one. Reported with the standard balanced error rate (BER, lower is better).

    net = train(epochs=8)                    # ISTD train split, 1330 images
    ber = evaluate(net)                      # ISTD test split, 540 images
    prob = predict(net, img_uint8)           # HxW shadow probability

Version 2 (`train_v2`) is the detector used for the T-ITS revision: SegFormer-B5 at 512 px, trained on
SBU (4,089 noisily labelled, mostly outdoor images) together with ISTD, with an augmentation that imitates
a low sun (warm white balance, gamma, low contrast, under/over exposure) and random resized crops. It is
selected on the SBU and ISTD *test* BER only; no KITTI or PandaSet frame is used for any choice.
"""
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
SIZE = 384


def _raw(img, size=SIZE):
    x = torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1).float()[None] / 255
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)


def _tensor(img, size=SIZE):
    return (_raw(img, size) - MEAN) / STD


def _mask(m, size=SIZE):
    m = torch.from_numpy((np.asarray(m) > 127).astype(np.float32))
    if m.ndim == 3:
        m = m[..., 0]
    return F.interpolate(m[None, None], size=(size, size), mode="nearest")[:, 0].long()


def sun_jitter(x, rng):
    """Photometric augmentation of a batch of RGB images in [0, 1] that imitates a low or a high sun.

    Warm or cool white balance (a low sun is orange, skylight in shadows is blue), gamma, contrast and
    exposure. It changes the colour of shadow and lit road together, so dark != shadow must be learned.
    """
    n = len(x)
    warm = torch.from_numpy(rng.uniform(-0.25, 0.25, n)).float()
    wb = torch.stack([1 + warm, torch.ones(n), 1 - warm], 1)[..., None, None]
    gamma = torch.from_numpy(np.exp(rng.uniform(np.log(0.6), np.log(1.6), n))).float()[:, None, None, None]
    contrast = torch.from_numpy(rng.uniform(0.6, 1.2, n)).float()[:, None, None, None]
    expo = torch.from_numpy(np.exp(rng.uniform(np.log(0.6), np.log(1.5), n))).float()[:, None, None, None]
    x = (x * wb * expo).clamp(1e-4, 1) ** gamma
    mean = x.mean((1, 2, 3), keepdim=True)
    return ((x - mean) * contrast + mean).clamp(0, 1)


def resized_crop(x, y, rng, min_scale=0.5):
    """Same random crop (area fraction min_scale..1) of images x (N3HW) and masks y (NHW), resized back."""
    _, _, h, w = x.shape
    s = np.sqrt(rng.uniform(min_scale, 1))
    ch, cw = int(h * s), int(w * s)
    i, j = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
    x = F.interpolate(x[:, :, i:i + ch, j:j + cw], size=(h, w), mode="bilinear", align_corners=False)
    y = F.interpolate(y[:, None, i:i + ch, j:j + cw].float(), size=(h, w), mode="nearest")[:, 0].long()
    return x, y


def load_istd(split):
    from datasets import load_dataset

    return load_dataset("Donghyun99/ISTD", split=split)


def build(backbone="nvidia/mit-b2"):
    from transformers import SegformerForSemanticSegmentation

    return SegformerForSemanticSegmentation.from_pretrained(backbone, num_labels=2).to(DEVICE)


def train(epochs=8, batch=8, lr=6e-5, backbone="nvidia/mit-b2", seed=0, log=print, limit=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = load_istd("train")
    if limit:
        data = data.select(range(limit))
    net = build(backbone).train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * (len(data) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    step = 0
    for ep in range(epochs):
        order = rng.permutation(len(data))
        running = []
        for k in range(0, len(order) - batch + 1, batch):
            rows = [data[int(i)] for i in order[k:k + batch]]
            x = torch.cat([_tensor(r["image"]) for r in rows])
            y = torch.cat([_mask(r["mask"]) for r in rows])
            if rng.random() < 0.5:  # horizontal flip
                x, y = x.flip(-1), y.flip(-1)
            # photometric jitter: shadows must not be "anything dark"
            x = x * (0.8 + 0.4 * torch.rand(len(x), 1, 1, 1)) + 0.2 * (torch.rand(len(x), 1, 1, 1) - 0.5)
            logits = net(pixel_values=x.to(DEVICE)).logits
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, y.to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            running.append(float(loss))
        log(f"epoch {ep + 1}/{epochs}  loss {np.mean(running):.4f}  ({step} steps)")
    return net.eval()


def load_sbu(root, split):
    """SBU Shadow (Vicente et al. 2016) from a folder tree with ShadowImages/ and ShadowMasks/ per split.

    Returns a list of dicts with lazily opened "image" and "mask", like the rows of the ISTD dataset.
    """
    from pathlib import Path

    from PIL import Image

    tag = {"train": "Train", "test": "Test"}[split]
    imgs = [p for p in Path(root).rglob("*") if p.is_file() and p.parent.name == "ShadowImages"
            and tag in str(p.parent.parent.name) and p.suffix.lower() in {".jpg", ".png"}]
    masks = {p.stem: p for p in Path(root).rglob("*") if p.is_file() and p.parent.name == "ShadowMasks"
             and tag in str(p.parent.parent.name)}
    return [_Lazy(p, masks[p.stem], Image) for p in sorted(imgs) if p.stem in masks]


class _Lazy:
    """One image/mask pair on disk, opened only when indexed (sample["image"], sample["mask"])."""

    def __init__(self, img, mask, Image):
        self._paths, self._Image = (img, mask), Image

    def __getitem__(self, k):
        p = self._paths[0] if k == "image" else self._paths[1]
        return self._Image.open(p).convert("RGB" if k == "image" else "L")


class Concat:
    """Several sample sequences (lists, Hugging Face datasets) seen as one."""

    def __init__(self, *parts):
        self.parts = parts
        self.offsets = np.cumsum([0] + [len(q) for q in parts])

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, i):
        j = int(np.searchsorted(self.offsets, i, side="right")) - 1
        return self.parts[j][int(i - self.offsets[j])]


def train_v2(samples, epochs=10, batch=8, lr=6e-5, backbone="nvidia/mit-b5", size=512, seed=0, log=print):
    """Train detector version 2 on a list of samples (dicts with "image" and "mask"), mixed precision."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = build(backbone).train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * (len(samples) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    step = 0
    for ep in range(epochs):
        order = rng.permutation(len(samples))
        running = []
        for k in range(0, len(order) - batch + 1, batch):
            rows = [samples[int(i)] for i in order[k:k + batch]]
            x = torch.cat([_raw(r["image"], size) for r in rows])
            y = torch.cat([_mask(r["mask"], size) for r in rows])
            if rng.random() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            x, y = resized_crop(x, y, rng)
            x = (sun_jitter(x, rng) - MEAN) / STD
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                logits = net(pixel_values=x.to(DEVICE)).logits
                logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
                yd = y.to(DEVICE)
                p = logits.float().softmax(1)[:, 1]
                dice = 1 - (2 * (p * yd).sum() + 1) / (p.sum() + yd.sum() + 1)
                loss = F.cross_entropy(logits.float(), yd) + 0.5 * dice
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            running.append(float(loss))
        log(f"epoch {ep + 1}/{epochs}  loss {np.mean(running):.4f}  ({step} steps)")
    return net.eval()


@torch.no_grad()
def predict(net, img, size=SIZE, tta=False):
    """HxW shadow probability; tta=True averages the prediction with that of the mirrored image."""
    h, w = np.asarray(img).shape[:2]
    x = _tensor(img, size).to(DEVICE)
    prob = net(pixel_values=x).logits.softmax(1)[:, 1:2]
    if tta:
        prob = 0.5 * (prob + net(pixel_values=x.flip(-1)).logits.softmax(1)[:, 1:2].flip(-1))
    return F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def ber(pred, gt):
    """Balanced error rate in %, the usual shadow-detection metric."""
    pos, neg = gt.sum(), (~gt).sum()
    tp, tn = (pred & gt).sum(), (~pred & ~gt).sum()
    return 100 * (1 - 0.5 * (tp / max(pos, 1) + tn / max(neg, 1)))


def evaluate(net, threshold=0.5, limit=None, data=None, size=SIZE, tta=False):
    """BER in % on the ISTD test split, or on `data` (a list of samples, e.g. load_sbu(root, "test"))."""
    if data is None:
        data = load_istd("test")
        if limit:
            data = data.select(range(limit))
    elif limit:
        data = data[:limit]
    tp = tn = pos = neg = 0
    for r in data:
        gt = np.asarray(r["mask"])
        gt = (gt[..., 0] if gt.ndim == 3 else gt) > 127
        pred = predict(net, r["image"], size, tta) > threshold
        tp += (pred & gt).sum()
        tn += (~pred & ~gt).sum()
        pos += gt.sum()
        neg += (~gt).sum()
    return float(100 * (1 - 0.5 * (tp / pos + tn / neg)))
