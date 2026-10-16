"""Pass 3: artistic cleanup of pass-2 strokes.

`freedom` (0..1) controls how far the drawing may drift from the photo in exchange
for cleaner, more graphic lines. All distances scale with the dog's size.
"""
import logging
import time

import cv2
import numpy as np
from scipy.interpolate import splev, splprep

log = logging.getLogger(__name__)


def dog_size(mask: np.ndarray) -> float:
    return float(np.sqrt(mask.sum()))


def path_length(p: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


# ---------- 3a: smooth curves ----------

def spline(p: np.ndarray, tol: float, step: float, closed: bool = False) -> np.ndarray:
    """Least-squares B-spline through the stroke; `tol` is the allowed wobble in px."""
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-3]
    p = p[keep]
    if closed and np.linalg.norm(p[0] - p[-1]) < 1e-3:
        p = p[:-1]
    if len(p) < 4:
        return p
    k = 3
    try:
        tck, _ = splprep([p[:, 0], p[:, 1]], s=len(p) * tol ** 2, k=k, per=int(closed))
    except Exception:  # degenerate strokes (all collinear/duplicate) keep their original shape
        return p
    n = max(8, int(path_length(p) / step))
    x, y = splev(np.linspace(0, 1, n), tck)
    out = np.stack([x, y], axis=1).astype(np.float32)
    return np.concatenate([out, out[:1]]) if closed else out


# ---------- 3b: simplify silhouette ----------

def simplify_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Shave off fur tufts thinner than ~2*radius, but put back big parts (legs, ears, tail)."""
    if radius < 1:
        return mask
    m = mask.astype(np.uint8)
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, disk(radius))
    lost = (m & ~opened.astype(bool)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lost, connectivity=8)
    big = np.zeros(n, bool)
    big[1:] = stats[1:, cv2.CC_STAT_AREA] > (radius * 6) ** 2
    out = opened.astype(bool) | big[labels]
    out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, disk(radius))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    if n > 1:
        out = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return out.astype(bool)


def outline_from_mask(mask: np.ndarray, tol: float, step: float) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        if len(c) > 20:
            out.append(spline(c[:, 0, :].astype(np.float32), tol, step, closed=True))
    return out


# ---------- 3c: remove clutter ----------

def runs(keep: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) index pairs of consecutive True runs."""
    edges = np.flatnonzero(np.diff(np.r_[0, keep.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def drop_frame_edges(outline: list, shape, margin: float) -> list:
    """Remove outline parts that just trace the photo border (dog cut off by the frame)."""
    h, w = shape
    out = []
    for q in outline:
        x, y = q[:, 0], q[:, 1]
        on_frame = (x < margin) | (y < margin) | (x > w - 1 - margin) | (y > h - 1 - margin)
        if not on_frame.any():
            out.append(q)
            continue
        if on_frame.all():
            continue
        # rotate so the stroke starts inside a frame run, then keep the off-frame pieces
        shift = int(np.flatnonzero(on_frame)[0])
        q, on_frame = np.roll(q[:-1], -shift, 0), np.roll(on_frame[:-1], -shift)
        out += [q[a:b] for a, b in runs(~on_frame) if b - a > 2]
    return out


def trim_hooks(p: np.ndarray, zone: float, max_turn: float = np.pi / 2) -> np.ndarray:
    """Cut off sharp hooks near either end of an open stroke."""
    if len(p) < 6:
        return p
    seg = np.diff(p, axis=0)
    ang = np.arctan2(seg[:, 1], seg[:, 0])
    turn = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)
    dist = np.r_[0, np.cumsum(np.linalg.norm(seg, axis=1))]
    total = dist[-1]
    start, end = 0, len(p)
    acc = 0.0
    for i in range(len(turn)):
        if dist[i + 1] > min(zone, total / 4):
            break
        acc += turn[i]
        if acc > max_turn:
            start, acc = i + 1, 0.0
    acc = 0.0
    for i in range(len(turn) - 1, -1, -1):
        if total - dist[i + 1] > min(zone, total / 4):
            break
        acc += turn[i]
        if acc > max_turn:
            end, acc = i + 2, 0.0
    return p[start:end] if end - start >= 3 else p


def drop_duplicates(inner: list, outline: list, shape, near: float, min_len: float) -> list:
    """Remove parts of strokes that run alongside the outline or a longer stroke."""
    canvas = np.zeros(shape, np.uint8)
    for q in outline:
        cv2.polylines(canvas, [np.round(q).astype(np.int32)], False, 1, 1)
    kept = []
    for q in sorted(inner, key=path_length, reverse=True):
        dist = cv2.distanceTransform(1 - canvas, cv2.DIST_L2, 3)
        xi = np.clip(np.round(q[:, 0]).astype(int), 0, shape[1] - 1)
        yi = np.clip(np.round(q[:, 1]).astype(int), 0, shape[0] - 1)
        free = dist[yi, xi] > near
        for a, b in runs(free):
            piece = q[a:b]
            if len(piece) > 2 and path_length(piece) >= min_len:
                kept.append(piece)
                cv2.polylines(canvas, [np.round(piece).astype(np.int32)], False, 1, 1)
    return kept


# ---------- 3d: clean feature shapes (eyes, nose, mouth) ----------

def tortuosity(p: np.ndarray) -> float:
    return path_length(p) / max(float(np.linalg.norm(p[-1] - p[0])), 1e-3)


def ellipse_points(center, axes, angle_deg, t: np.ndarray) -> np.ndarray:
    a, b = axes[0] / 2, axes[1] / 2
    th = np.deg2rad(angle_deg)
    x, y = a * np.cos(t), b * np.sin(t)
    return np.stack([center[0] + x * np.cos(th) - y * np.sin(th),
                     center[1] + x * np.sin(th) + y * np.cos(th)], axis=1).astype(np.float32)


def fit_shape(p: np.ndarray, max_err: float, step: float) -> np.ndarray | None:
    """Replace a curly stroke with a clean ellipse (or the arc of one it covers)."""
    if len(p) < 5:
        return None
    (cx, cy), (w, h), ang = cv2.fitEllipse(p.reshape(-1, 1, 2))
    if min(w, h) < 2 or max(w, h) / min(w, h) > 4:
        return None
    th = np.deg2rad(ang)
    dx, dy = p[:, 0] - cx, p[:, 1] - cy
    u = (dx * np.cos(th) + dy * np.sin(th)) / (w / 2)
    v = (-dx * np.sin(th) + dy * np.cos(th)) / (h / 2)
    if np.abs(np.hypot(u, v) - 1).mean() > max_err:
        return None

    bins = np.zeros(36, bool)
    bins[((np.arctan2(v, u) + np.pi) / (2 * np.pi) * 36).astype(int) % 36] = True
    n = max(24, int(np.pi * (w + h) / 2 / step))
    if bins.mean() > 0.75:
        t = np.linspace(0, 2 * np.pi, n)
    else:
        # longest run of covered angle bins (wrapping around)
        doubled = np.r_[bins, bins]
        best, start, cur = 0, 0, 0
        for i, b in enumerate(doubled):
            cur = cur + 1 if b else 0
            if cur > best and cur <= 36:
                best, start = cur, i - cur + 1
        t = np.linspace(start, start + best, max(8, int(n * best / 36))) / 36 * 2 * np.pi - np.pi
    return ellipse_points((cx, cy), (w, h), ang, t)


def clean_features(inner: list, size: float, freedom: float, step: float) -> list:
    out = []
    small = 0.12 * size
    for q in inner:
        span = float(np.ptp(q, axis=0).max())
        curly = tortuosity(q) > 1.6 or np.linalg.norm(q[-1] - q[0]) < 0.2 * path_length(q)
        if span < small and curly:
            shape = fit_shape(q, max_err=0.1 + 0.2 * freedom, step=step)
            if shape is not None:
                out.append(shape)
                continue
            q = spline(q, tol=0.01 * size * freedom, step=step)
            if float(np.ptp(q, axis=0).max()) < 0.04 * size * freedom:
                continue  # small squiggle that isn't a clean shape: drop it
        out.append(q)
    return out


# ---------- 3e: join line ends to the outline ----------

def snap_to_outline(inner: list, outline: list, reach: float) -> list:
    if not outline or reach <= 0:
        return inner
    pts = np.concatenate(outline)
    out = []
    for q in inner:
        if len(q) < 3:
            out.append(q)
            continue
        ends = []
        for end, prev in ((q[0], q[1]), (q[-1], q[-2])):
            d = np.linalg.norm(pts - end, axis=1)
            i = int(np.argmin(d))
            heading = (end - prev) / max(float(np.linalg.norm(end - prev)), 1e-6)
            toward = (pts[i] - end) / max(float(d[i]), 1e-6)
            ends.append(pts[i] if 0 < d[i] < reach and heading @ toward > 0.3 else None)
        if ends[0] is not None:
            q = np.vstack([ends[0][None], q])
        if ends[1] is not None:
            q = np.vstack([q, ends[1][None]])
        out.append(q)
    return out


# ---------- entry point ----------

def refine(inner: list, outline: list, blobs: list, mask: np.ndarray, freedom: float):
    """Returns cleaned (inner, outline, blobs)."""
    if freedom <= 0:
        return inner, outline, blobs
    t = time.perf_counter()
    size = dog_size(mask)
    step = max(1.0, 0.002 * size)

    simple = simplify_mask(mask, int(0.012 * size * freedom))
    outline = outline_from_mask(simple, tol=0.004 * size * freedom, step=step)
    outline = drop_frame_edges(outline, mask.shape, margin=0.004 * size)
    inner = [spline(q, tol=0.003 * size * freedom, step=step) for q in inner]
    inner = [trim_hooks(q, zone=0.04 * size) for q in inner]
    inner = drop_duplicates(inner, outline, mask.shape, near=0.012 * size * (0.5 + freedom),
                            min_len=0.03 * size)
    inner = clean_features(inner, size, freedom, step)
    inner = snap_to_outline(inner, outline, reach=0.03 * size * freedom)

    log.info("  pass3: %d stroke(s) + %d outline, freedom %.2f in %.1fs",
             len(inner), len(outline), freedom, time.perf_counter() - t)
    return inner, outline, blobs
