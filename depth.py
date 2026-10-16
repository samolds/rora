"""Monocular depth of the dog (Depth Anything V2) and "form lines" derived from it.

Depth jumps inside the silhouette are overlap lines (leg in front of chest, ear in front of
head); sharp bends in the surface mark form (shoulder, haunch). Both give a drawing volume.
"""
import logging
import time
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

import env  # noqa: F401  (keeps model downloads in ./models)

log = logging.getLogger(__name__)

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


@lru_cache(maxsize=None)
def load_model(device: str):
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    t = time.perf_counter()
    proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).eval()
    try:
        model.to(device)
    except Exception as e:  # fall back if the GPU backend lacks an op
        log.info("  depth model stays on cpu (%s)", e)
        device = "cpu"
    log.info("  loaded depth model in %.1fs on %s", time.perf_counter() - t, device)
    return proc, model, device, torch


def estimate(rgb: np.ndarray, mask: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Relative depth in [0,1] (1 = nearest) over the dog, same size as the image."""
    proc, model, device, torch = load_model(device)
    t = time.perf_counter()
    ys, xs = np.nonzero(mask)
    pad = int(0.05 * max(np.ptp(ys), np.ptp(xs)))
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)
    crop = rgb[y0:y1, x0:x1]
    inputs = proc(images=Image.fromarray(crop), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        pred = model(**inputs).predicted_depth
    d = torch.nn.functional.interpolate(pred[:, None], size=crop.shape[:2], mode="bicubic",
                                        align_corners=False)[0, 0].cpu().numpy()
    out = np.zeros(mask.shape, np.float32)
    out[y0:y1, x0:x1] = d
    vals = out[mask]
    lo, hi = np.percentile(vals, [1, 99])
    out = (np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1) * mask).astype(np.float32)
    log.info("  depth in %.1fs", time.perf_counter() - t)
    return out


def form_map(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Soft line map in [0,1]: depth jumps (overlaps) plus strong surface bends."""
    size = float(np.sqrt(mask.sum()))
    d = cv2.GaussianBlur(depth, (0, 0), max(1.0, 0.004 * size))
    gx, gy = cv2.Sobel(d, cv2.CV_32F, 1, 0), cv2.Sobel(d, cv2.CV_32F, 0, 1)
    jump = np.hypot(gx, gy)
    bend = np.abs(cv2.Laplacian(cv2.GaussianBlur(d, (0, 0), max(1.0, 0.01 * size)), cv2.CV_32F))
    # stay clear of the silhouette, where depth drops to the (masked-out) background
    r = max(3, int(0.02 * size))
    inner = cv2.erode(mask.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)

    def norm(m):
        hi = np.percentile(m[inner], 99) if inner.any() else m.max()
        return np.clip(m / max(hi, 1e-6), 0, 1)

    return (np.maximum(norm(jump), 0.6 * norm(bend)) * inner).astype(np.float32)


def form_strokes(form: np.ndarray, mask: np.ndarray, count: int, exclude: np.ndarray | None,
                 coverage: float = 0.04) -> list[np.ndarray]:
    """The `count` strongest, longest form lines, as smooth polylines."""
    import cleanup
    import strokes
    from skimage.morphology import skeletonize

    if count <= 0:
        return []
    size = float(np.sqrt(mask.sum()))
    vals = form[mask]
    thr = max(0.25, float(np.percentile(vals, 100 * (1 - coverage)))) if vals.size else 1.0
    skel = skeletonize(form > thr)
    paths = [q for q in strokes.trace(skel) if strokes.path_length(q) >= 0.01 * size]
    paths = strokes.join_paths(paths, gap=0.012 * size)
    paths = [strokes.smooth(q, 0.004 * size) for q in paths]
    paths = [q for q in paths if strokes.path_length(q) >= 0.06 * size]
    # lines hugging the silhouette are just the body curving away (a "double outline")
    r = int(0.035 * size)
    near_edge = ~cv2.erode(mask.astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)
    exclude = near_edge if exclude is None else (exclude | near_edge)
    paths = cleanup.clip_paths(paths, ~exclude, 0.05 * size)
    return strokes.rank(paths, form, count, min_score=0.05)
