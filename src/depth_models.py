"""Metric depth models used by the shadow study, one predict(img) -> HxW metres interface.

The pilot used Depth Anything V2 Small; the study compares it with stronger models:
DA-V2 Large (pixels only), Depth Pro and MoGe-2 Large (both predict their own camera).
"""
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HF:
    def __init__(self, repo, kind):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.kind = kind
        self.last_focal = None  # focal length in pixels, for models that predict their camera
        self.proc = AutoImageProcessor.from_pretrained(repo)
        self.model = AutoModelForDepthEstimation.from_pretrained(repo).eval().to(DEVICE)

    @torch.no_grad()
    def __call__(self, img):
        h, w = img.shape[:2]
        out = self.model(**self.proc(images=img, return_tensors="pt").to(DEVICE))
        if self.kind == "dav2":
            d = out.predicted_depth
        else:
            res = self.proc.post_process_depth_estimation(out, target_sizes=[(h, w)])[0]
            d = res["predicted_depth"]
            f = res.get("focal_length")
            self.last_focal = None if f is None else float(f)
        d = d.float().cpu()
        while d.ndim < 4:
            d = d[None]
        return F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()


class MoGe:
    def __init__(self, repo):
        from moge.model.v2 import MoGeModel

        self.model = MoGeModel.from_pretrained(repo).eval().to(DEVICE)
        self.last_focal = None

    @torch.no_grad()
    def __call__(self, img):
        x = torch.from_numpy(img).permute(2, 0, 1).float().to(DEVICE) / 255
        out = self.model.infer(x, use_fp16=DEVICE.type == "cuda")
        d = out["depth"].float().cpu().numpy()
        # MoGe intrinsics are normalised by the image size
        self.last_focal = float(out["intrinsics"][0, 0]) * img.shape[1]
        d[~out["mask"].cpu().numpy()] = 0
        return np.nan_to_num(d, nan=0.0, posinf=0.0)


MODELS = {
    "dav2-s": lambda: HF("depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf", "dav2"),
    "dav2-l": lambda: HF("depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf", "dav2"),
    "depthpro": lambda: HF("apple/DepthPro-hf", "depthpro"),
    "moge2-l": lambda: MoGe("Ruicheng/moge-2-vitl-normal"),
}
