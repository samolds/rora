"""--square: crop a clip to a square that slides along its long side to keep the subjects'
faces in frame (the middle of their eyes and nose, found with OWLv2 as for the tattoo looks)."""
import numpy as np

from .features import FEATURE_QUERIES, nms
from .segment import Subject

MIN_FEATURE_SCORE = 0.2
HEAD_FRACTION = 0.45  # no head found: take the top of the subject's box as its head
SMOOTH_SECONDS = 0.35


def face_center(finder, rgb: np.ndarray, subject: Subject) -> tuple[float, float] | None:
    """(x, y) of the middle of the subject's eyes and nose, else of its head."""
    if not subject.mask.any():
        return None
    head = finder.find_head(rgb, subject.mask, subject.kind)
    if head is None:
        x0, y0, x1, y1 = subject.box
        return (x0 + x1) / 2, y0 + HEAD_FRACTION / 2 * (y1 - y0)
    hx0, hy0, hx1, hy1 = (int(round(v)) for v in head.box)
    if hx1 - hx0 < 4 or hy1 - hy0 < 4:
        return tuple(map(float, head.center))
    queries = {k: FEATURE_QUERIES[subject.kind][k] for k in ("eye", "nose")}
    found = [d for d in nms(finder.detect(rgb[hy0:hy1, hx0:hx1], queries))
             if d.score >= MIN_FEATURE_SCORE]
    best = [next((d for d in found if d.name == k), None) for k in ("eye", "nose")]
    points = [d.center for d in best if d is not None]
    if not points:
        return tuple(map(float, head.center))
    x, y = np.mean(points, axis=0) + (hx0, hy0)
    return float(x), float(y)


def offsets(centers: list[list[tuple[float, float]]], shape: tuple, smooth: float) -> list[int]:
    """Where the square starts along the frame's long side, per frame: centred on the mean of
    that frame's face centres, filled in across frames with none, smoothed with a Gaussian of
    `smooth` frames (so it glides rather than jumps) and kept inside the frame."""
    h, w = shape[:2]
    n, side, span = len(centers), min(h, w), abs(h - w)
    axis = 1 if h > w else 0
    known = [(i, float(np.mean([c[axis] for c in cs]))) for i, cs in enumerate(centers) if cs]
    if not span or not known:
        return [span // 2] * n
    pos = np.interp(np.arange(n), *zip(*known))
    if smooth > 0:
        r = int(np.ceil(3 * smooth))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / smooth) ** 2)
        pos = np.convolve(np.pad(pos, r, mode="edge"), k / k.sum(), mode="valid")
    return np.clip(np.round(pos - side / 2), 0, span).astype(int).tolist()


def crop(image: np.ndarray, start: int) -> np.ndarray:
    """The square of `image` (RGB frame or mask) starting `start` px along its long side."""
    h, w = image.shape[:2]
    side = min(h, w)
    return image[start:start + side] if h > w else image[:, start:start + side]


def crop_subject(s: Subject, start: int) -> Subject:
    h, w = s.mask.shape
    side = min(h, w)
    x0, y0, x1, y1 = s.box
    if h > w:
        box = (x0, max(0, y0 - start), x1, min(side, y1 - start))
    else:
        box = (max(0, x0 - start), y0, min(side, x1 - start), y1)
    return Subject(np.ascontiguousarray(crop(s.mask, start)), box, s.score, s.kind)
