"""Pass 1: draw the dog many different ways, as soft line maps in [0,1].

Each map disagrees about fur texture but agrees about real features (eyes, nose,
mouth, ears, toes), which is what pass 2 relies on.
"""
import logging
import time
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

import env  # noqa: F401  (keeps model downloads in ./models)

log = logging.getLogger(__name__)

ML_MODELS = {
    "hed": ("HEDdetector", {"safe": True}),
    "pidinet": ("PidiNetDetector", {"safe": True}),
    "lineart": ("LineartDetector", {"coarse": True}),
    "anime": ("LineartAnimeDetector", {}),
}


def isolate(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Dog on a flat background of its own average colour, so no background edges leak in."""
    out = rgb.copy()
    out[~mask] = rgb[mask].mean(axis=0).astype(np.uint8)
    return out


def normalize(m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = m.astype(np.float32)
    hi = np.percentile(m[mask], 99.5) if mask.any() else m.max()
    return np.clip(m / max(hi, 1e-6), 0, 1)


def flatten(rgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Edge-preserving smoothing that melts fur strands into flat regions."""
    h, w = rgb.shape[:2]
    f = 900 / max(h, w)
    small = cv2.resize(rgb, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    bgr = cv2.pyrMeanShiftFiltering(bgr, int(8 * strength), int(24 * strength))
    small = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


# ---------- classical maps ----------

def map_xdog(flat, mask):
    g = cv2.cvtColor(flat, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    s = max(flat.shape[:2]) / 900
    d = cv2.GaussianBlur(g, (0, 0), 1.2 * s) - cv2.GaussianBlur(g, (0, 0), 2.0 * s)
    return normalize(np.maximum(-d, 0), mask)


def map_canny(flat, mask):
    g = cv2.cvtColor(flat, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), max(flat.shape[:2]) / 900)
    e = cv2.Canny(g, 30, 80).astype(np.float32)
    return normalize(cv2.GaussianBlur(e, (0, 0), 1.5), mask)


def map_regions(flat, mask, k: int = 5):
    """Borders between flat colour regions (k-means on the smoothed dog)."""
    h, w = mask.shape
    f = 900 / max(h, w)
    size = (round(w * f), round(h * f))
    small = cv2.resize(flat, size, interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    sample = lab[small_mask.ravel()]
    _, _, centers = cv2.kmeans(sample, k, None,
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                               3, cv2.KMEANS_PP_CENTERS)
    labels = np.argmin(((lab[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
    labels = cv2.medianBlur(labels.reshape(small_mask.shape).astype(np.uint8), 5)
    labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
    e = cv2.Canny(labels * (255 // k), 1, 1).astype(np.float32)
    return normalize(cv2.GaussianBlur(e, (0, 0), 1.5), mask)


# ---------- pretrained maps ----------

@lru_cache(maxsize=None)
def load_model(name: str, device: str):
    import controlnet_aux

    cls_name, _ = ML_MODELS[name]
    t = time.perf_counter()
    model = getattr(controlnet_aux, cls_name).from_pretrained("lllyasviel/Annotators")
    try:
        model = model.to(device)
    except Exception as e:  # some annotators only run on cpu
        log.info("  %s stays on cpu (%s)", name, e)
    log.info("  loaded %s in %.1fs", name, time.perf_counter() - t)
    return model


def map_ml(name: str, iso: np.ndarray, mask: np.ndarray, device: str, res: int = 1024):
    _, kwargs = ML_MODELS[name]
    out = load_model(name, device)(Image.fromarray(iso), detect_resolution=res,
                                   image_resolution=res, output_type="np", **kwargs)
    out = out.mean(axis=2) if out.ndim == 3 else out
    out = cv2.resize(out.astype(np.float32), mask.shape[::-1], interpolation=cv2.INTER_LINEAR)
    return normalize(out, mask)


def candidate_maps(rgb: np.ndarray, mask: np.ndarray, device: str = "cpu",
                   use_ml: bool = True) -> dict[str, np.ndarray]:
    iso = isolate(rgb, mask)
    flat = flatten(iso)
    maps = {}
    for name, fn in (("xdog", map_xdog), ("canny", map_canny), ("regions", map_regions)):
        t = time.perf_counter()
        maps[name] = fn(flat, mask)
        log.info("  pass1 %-8s %.1fs", name, time.perf_counter() - t)
    if use_ml:
        for name in ML_MODELS:
            t = time.perf_counter()
            maps[name] = map_ml(name, iso, mask, device)
            log.info("  pass1 %-8s %.1fs", name, time.perf_counter() - t)
    return {k: v * mask for k, v in maps.items()}
