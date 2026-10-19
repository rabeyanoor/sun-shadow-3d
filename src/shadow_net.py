"""A learned shadow detector: SegFormer fine-tuned on ISTD (shadow image + shadow mask).

The pilot used a dark-asphalt threshold, and the synthetic study (notebook 04) showed that
this is where the shape/focal signal got lost. Hugging Face has no ready shadow detector,
so we train one. Reported with the standard balanced error rate (BER, lower is better).

    net = train(epochs=8)                    # ISTD train split, 1330 images
    ber = evaluate(net)                      # ISTD test split, 540 images
    prob = predict(net, img_uint8)           # HxW shadow probability
"""
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
SIZE = 384


def _tensor(img, size=SIZE):
    x = torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1).float()[None] / 255
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return (x - MEAN) / STD


def _mask(m, size=SIZE):
    m = torch.from_numpy((np.asarray(m) > 127).astype(np.float32))
    if m.ndim == 3:
        m = m[..., 0]
    return F.interpolate(m[None, None], size=(size, size), mode="nearest")[:, 0].long()


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


@torch.no_grad()
def predict(net, img):
    h, w = np.asarray(img).shape[:2]
    logits = net(pixel_values=_tensor(img).to(DEVICE)).logits
    prob = logits.softmax(1)[:, 1:2]
    return F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def ber(pred, gt):
    """Balanced error rate in %, the usual shadow-detection metric."""
    pos, neg = gt.sum(), (~gt).sum()
    tp, tn = (pred & gt).sum(), (~pred & ~gt).sum()
    return 100 * (1 - 0.5 * (tp / max(pos, 1) + tn / max(neg, 1)))


def evaluate(net, threshold=0.5, limit=None):
    data = load_istd("test")
    if limit:
        data = data.select(range(limit))
    tp = tn = pos = neg = 0
    for r in data:
        gt = np.asarray(r["mask"])
        gt = (gt[..., 0] if gt.ndim == 3 else gt) > 127
        pred = predict(net, r["image"]) > threshold
        tp += (pred & gt).sum()
        tn += (~pred & ~gt).sum()
        pos += gt.sum()
        neg += (~gt).sum()
    return float(100 * (1 - 0.5 * (tp / pos + tn / neg)))
