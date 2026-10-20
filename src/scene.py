"""Scene helpers: semantic masks (road / sidewalk / sky) and observed shadows on the road."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

# cityscapes train ids
ROAD, SIDEWALK, TERRAIN, SKY = 0, 1, 9, 10

_SEG_ID = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
_seg_model = None
_seg_proc = None


@torch.no_grad()
def segment(img):
    global _seg_model, _seg_proc
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    if _seg_model is None:
        _seg_proc = SegformerImageProcessor.from_pretrained(_SEG_ID)
        _seg_model = SegformerForSemanticSegmentation.from_pretrained(_SEG_ID).eval()
    inputs = _seg_proc(images=img, return_tensors="pt")
    logits = _seg_model(**inputs).logits
    logits = F.interpolate(logits, size=img.shape[:2], mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].numpy()


_m2f = None


@torch.no_grad()
def segment_m2f(img, repo="facebook/mask2former-swin-large-cityscapes-semantic"):
    """Stronger Cityscapes segmentation (Mask2Former Swin-L), same train ids as segment()."""
    global _m2f
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    if _m2f is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _m2f = (AutoImageProcessor.from_pretrained(repo),
                Mask2FormerForUniversalSegmentation.from_pretrained(repo).eval().to(dev), dev)
    proc, model, dev = _m2f
    out = model(**proc(images=img, return_tensors="pt").to(dev))
    return proc.post_process_semantic_segmentation(out, target_sizes=[img.shape[:2]])[0].cpu().numpy()


def receiver_mask(labels, classes=(ROAD,)):
    """Flat, mostly uniform surfaces where a cast shadow is easy to see.

    Road only by default: in notebook 02 the tram-track gravel was labelled sidewalk and
    its dark stones were read as shadow.
    """
    return np.isin(labels, list(classes))


def observed_shadow(img, receiver, dark_ratio=0.55):
    """Shadow pixels on the receiver: much darker than the sunlit part of the same surface.

    This is a deliberately simple detector. It only makes sense on albedo-neutral
    receivers like asphalt, which is why we restrict it to road / sidewalk.
    """
    lum = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    lit_level = np.percentile(lum[receiver], 80)
    dark = (lum < dark_ratio * lit_level) & receiver
    dark = ndimage.binary_opening(dark, iterations=1)
    return ndimage.median_filter(dark.astype(np.uint8), size=5).astype(bool) & receiver


def observed_shadow_net(prob, receiver, threshold=0.5):
    """Shadow pixels on the receiver from a learned detector's probability map."""
    sh = ndimage.median_filter((prob > threshold).astype(np.uint8), size=5).astype(bool)
    return sh & receiver


def densify_lidar(sparse, max_gap=2):
    """Fill small holes in a sparse depth map with the nearest lidar point; leave big holes empty."""
    has = sparse > 0
    dist, idx = ndimage.distance_transform_edt(~has, return_indices=True)
    dense = sparse[idx[0], idx[1]]
    dense[dist > max_gap] = 0
    return dense


def iou(a, b, region):
    a, b = a & region, b & region
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else float("nan")


def kappa(a, b, region):
    """Cohen's kappa of two masks on the region: IoU-like agreement corrected for chance.

    IoU rewards a larger predicted mask when the target covers a sizeable part of the region;
    kappa subtracts the agreement two independent masks of the same coverage would reach.
    """
    a, b = a[region], b[region]
    if a.size == 0:
        return float("nan")
    p_a, p_b = a.mean(), b.mean()
    observed = (a == b).mean()
    chance = p_a * p_b + (1 - p_a) * (1 - p_b)
    return float((observed - chance) / (1 - chance)) if chance < 1 else float("nan")
