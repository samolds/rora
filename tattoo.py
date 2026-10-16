"""Minimalist tattoo-style line art.

Pass 1 (edgemaps) draws the dog many ways; pass 2 (strokes) keeps only lines most of
them agree on; pass 3 (cleanup) redraws them as clean, graphic strokes. The same
strokes are then drawn in different line looks.
"""
import logging

import cv2
import numpy as np

import cleanup
import depth
import edgemaps
import facedraw
import features
import strokes
from segment import pick_device
from styles import STYLES, register, scale

log = logging.getLogger(__name__)

PASS2 = dict(min_votes=3.5, tolerance=6, coverage=0.06, edge_band=12, min_length=50,
             join_gap=14, smooth=2.5, max_strokes=-1, min_score=0.05, max_blobs=0, darkness=0.15,
             freedom=0.7, features=1, head_strokes=-1, nose_solid=0, almond_eyes=0, pupils=0,
             form_lines=-1)
NAMES: list[str] = []

# Priority: the outline and face features are always drawn. --detail only sets how many
# extra lines are added: form lines from depth (overlaps like leg-over-chest, which give
# volume), body lines (legs, ears, folds) and secondary head lines (brow, muzzle).
FORM_LINES_AT_FULL_DETAIL = 10
BODY_STROKES_AT_FULL_DETAIL = 24
HEAD_STROKES_AT_FULL_DETAIL = 10

_cache: dict = {}
_finder = None


def get_finder(device: str):
    global _finder
    if _finder is None:
        _finder = features.FeatureFinder(device)
    return _finder


def head_regions(mask, face, size):
    """(head window, head core, feature zone) as boolean images."""
    x0, y0, x1, y1 = features.head_window(face, mask.shape)
    in_head = np.zeros(mask.shape, bool)
    in_head[y0:y1, x0:x1] = True
    # body lines reach a bit into the head so lines continue across the seam;
    # pass 3's duplicate removal keeps the longer of any overlapping pair
    m = int(0.06 * max(x1 - x0, y1 - y0))
    head_core = np.zeros(mask.shape, bool)
    head_core[y0 + m:y1 - m, x0 + m:x1 - m] = True
    return in_head, head_core, features.feature_zone(face, mask.shape, grow=0.01 * size)


def get_strokes(rgb, mask, p):
    """Passes 1-3 (+ head pass), cached so several tattoo styles on one dog share the work."""
    key = (rgb.shape, int(mask.sum()), hash(rgb[::37, ::37].tobytes()))
    if _cache.get("key") != key:
        _cache.clear()
        _cache["key"] = key
    device = pick_device(p.get("device", "auto"))
    if "maps" not in _cache:
        _cache["maps"] = edgemaps.candidate_maps(rgb, mask, device)

    face = None
    if p["features"]:
        if "face" not in _cache:
            _cache["face"] = get_finder(device).find_features(rgb, mask)
            if _cache["face"] is not None:
                _cache["head_maps"] = features.head_maps(rgb, mask, _cache["face"], device)
        face = _cache["face"]

    q = {k: p[k] for k in PASS2}
    if q["max_strokes"] < 0:
        # without a found face, interior lines are the only way eyes/nose show up
        floor = 0 if face is not None else 6
        q["max_strokes"] = floor + round(BODY_STROKES_AT_FULL_DETAIL * p["detail"])
    if q["head_strokes"] < 0:
        q["head_strokes"] = round(HEAD_STROKES_AT_FULL_DETAIL * p["detail"])
    if q["form_lines"] < 0:
        q["form_lines"] = round(FORM_LINES_AT_FULL_DETAIL * p["detail"])
    skey = ("strokes", tuple(sorted(q.items())))
    if skey not in _cache:
        size = cleanup.dog_size(mask)
        feature_lines = []
        if face is not None:
            in_head, head_core, zone = head_regions(mask, face, size)
            inner, outline, blobs, _ = strokes.extract(_cache["maps"], mask, rgb, q,
                                                       exclude=head_core | zone)
            head = features.head_strokes(_cache["head_maps"], {**q, "max_strokes": q["head_strokes"]})
            head = cleanup.clip_paths(head, in_head & ~zone, 0.01 * size)
            log.info("  lines: %d body + %d head (detail %.2f)", len(inner), len(head), p["detail"])
            inner = inner + head
        else:
            inner, outline, blobs, _ = strokes.extract(_cache["maps"], mask, rgb, q)
        if q["form_lines"] > 0:
            if "form" not in _cache:
                _cache["form"] = depth.form_map(depth.estimate(rgb, mask, device), mask)
            form = depth.form_strokes(_cache["form"], mask, q["form_lines"],
                                      exclude=zone if face is not None else None)
            log.info("  form lines: %d from depth", len(form))
            inner = form + inner  # listed first so ties in duplicate removal favour form lines
        inner, outline, blobs = cleanup.refine(inner, outline, blobs, mask, q["freedom"])
        if face is not None:
            inner = cleanup.clip_paths(inner, ~zone, 0.02 * size)  # smoothing may drift into features
            feature_lines, feature_fills = facedraw.draw_face(face, rgb, mask, size, bool(q["nose_solid"]),
                                                              bool(q["almond_eyes"]), bool(q["pupils"]))
            blobs = blobs + feature_fills
        _cache[skey] = (inner + feature_lines, outline, blobs)
    return _cache[skey]


# ---------- drawing helpers (on a 2x supersampled canvas) ----------

SS = 2


def resample(pts: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    d = np.concatenate([[0], np.cumsum(seg)])
    if d[-1] < step:
        return pts
    t = np.arange(0, d[-1], step)
    return np.stack([np.interp(t, d, pts[:, 0]), np.interp(t, d, pts[:, 1])], axis=1)


def draw_line(canvas, pts, width):
    """Variable-width polyline with round caps. `width` is a scalar or per-point array."""
    pts = pts * SS
    widths = np.broadcast_to(np.asarray(width, np.float32) * SS, (len(pts),))
    for i in range(len(pts) - 1):
        w = max(1, int(round((widths[i] + widths[i + 1]) / 2)))
        cv2.line(canvas, tuple(np.round(pts[i]).astype(int)), tuple(np.round(pts[i + 1]).astype(int)),
                 1, w, cv2.LINE_8)


def draw_dots(canvas, pts, radius, spacing):
    for x, y in resample(pts, spacing) * SS:
        cv2.circle(canvas, (int(round(x)), int(round(y))), max(1, int(round(radius * SS))), 1, -1)


def fill_blobs(canvas, blobs):
    """A blob is a polygon, or a list [outer, hole, ...] where holes are cut back out."""
    for b in blobs:
        parts = b if isinstance(b, list) else [b]
        for i, poly in enumerate(parts):
            cv2.fillPoly(canvas, [np.round(poly * SS).astype(np.int32)], 0 if i else 1)


def finish(canvas, shape) -> np.ndarray:
    small = cv2.resize(canvas.astype(np.float32), shape[::-1], interpolation=cv2.INTER_AREA)
    return small > 0.5


def new_canvas(mask):
    return np.zeros((mask.shape[0] * SS, mask.shape[1] * SS), np.uint8)


def smooth_noise(n: int, rng, period: float) -> np.ndarray:
    """1-D low-frequency noise in [-1, 1] along a stroke."""
    k = max(2, int(n / max(period, 1)) + 2)
    knots = rng.uniform(-1, 1, k)
    return np.interp(np.linspace(0, k - 1, n), np.arange(k), knots)


# ---------- the line looks ----------

def tattoo_style(name, description, **defaults):
    def wrap(fn):
        register(name, description, **{**PASS2, **defaults})(fn)
        STYLES[name].clip = False  # the outline stroke straddles the mask edge
        NAMES.append(name)
        return fn
    return wrap


@tattoo_style("fineline", "Tattoo: thin, even single-needle line", weight=3)
def fineline(rgb, mask, p):
    inner, outline, blobs = get_strokes(rgb, mask, p)
    w = p["weight"] * scale(rgb)
    c = new_canvas(mask)
    for q in inner + outline:
        draw_line(c, q, w)
    fill_blobs(c, blobs)
    return finish(c, mask.shape)


@tattoo_style("bold", "Tattoo: thick, even line with a heavier outline (traditional)",
              weight=9, outline_ratio=1.5, nose_solid=1)
def bold(rgb, mask, p):
    inner, outline, blobs = get_strokes(rgb, mask, p)
    w = p["weight"] * scale(rgb)
    c = new_canvas(mask)
    for q in outline:
        draw_line(c, q, w * p["outline_ratio"])
    for q in inner:
        draw_line(c, q, w)
    fill_blobs(c, blobs)
    return finish(c, mask.shape)


@tattoo_style("brush", "Tattoo: tapered strokes, thin at the ends like an ink brush",
              weight=10, taper=0.6, seed=1)
def brush(rgb, mask, p):
    inner, outline, blobs = get_strokes(rgb, mask, p)
    s = scale(rgb)
    w = p["weight"] * s
    rng = np.random.default_rng(int(p["seed"]))
    c = new_canvas(mask)
    for q in inner:
        q = resample(q, 2 * s)
        t = np.linspace(0, 1, len(q))
        draw_line(c, q, np.maximum(w * np.sin(np.pi * t) ** p["taper"], 1.0 * s))
    for q in outline:
        q = resample(q, 2 * s)
        pressure = 0.65 + 0.35 * smooth_noise(len(q), rng, period=300 * s)
        draw_line(c, q, w * pressure)
    fill_blobs(c, blobs)
    return finish(c, mask.shape)


@tattoo_style("sketchy", "Tattoo: loose, hand-drawn double strokes", weight=3, passes=2,
              wobble=4, seed=2)
def sketchy(rgb, mask, p):
    inner, outline, blobs = get_strokes(rgb, mask, p)
    s = scale(rgb)
    w = p["weight"] * s
    rng = np.random.default_rng(int(p["seed"]))
    c = new_canvas(mask)
    for q in inner + outline:
        q = resample(q, 3 * s)
        if len(q) < 3:
            continue
        tangent = np.gradient(q, axis=0)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], 1)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
        for _ in range(int(p["passes"])):
            off = p["wobble"] * s * smooth_noise(len(q), rng, period=120 * s)
            shifted = q + normal * off[:, None]
            # overshoot ends slightly, like a quick pen flick
            ext = max(1, len(q) // 12)
            draw_line(c, shifted[rng.integers(0, ext):len(q) - rng.integers(0, ext)], w)
    fill_blobs(c, blobs)
    return finish(c, mask.shape)


@tattoo_style("dotwork", "Tattoo: lines made of evenly spaced dots", weight=6, spacing=2.2)
def dotwork(rgb, mask, p):
    inner, outline, blobs = get_strokes(rgb, mask, p)
    r = p["weight"] * scale(rgb) / 2
    c = new_canvas(mask)
    for q in outline:
        draw_dots(c, q, r * 1.25, r * 1.25 * p["spacing"] * 2)
    for q in inner:
        draw_dots(c, q, r, r * p["spacing"] * 2)
    fill_blobs(c, blobs)
    return finish(c, mask.shape)


@tattoo_style("minimal", "Tattoo: fewest strokes, outline with negative-space gaps",
              weight=4, detail=0.2, gaps=6, gap_size=0.025, seed=3)
def minimal(rgb, mask, p):
    inner, outline, blobs = get_strokes(rgb, mask, p)
    s = scale(rgb)
    w = p["weight"] * s
    rng = np.random.default_rng(int(p["seed"]))
    c = new_canvas(mask)
    for q in inner:
        draw_line(c, q, w)
    for q in outline:
        q = resample(q, 2 * s)
        keep = np.ones(len(q), bool)
        gap = int(len(q) * p["gap_size"])
        for start in rng.choice(len(q), int(p["gaps"]), replace=False):
            keep[start:start + gap] = False
        # draw each kept run separately so gaps stay open
        idx = np.flatnonzero(np.diff(np.concatenate([[0], keep.astype(int), [0]])))
        for a, b in zip(idx[::2], idx[1::2]):
            if b - a > 1:
                draw_line(c, q[a:b], w)
    fill_blobs(c, blobs)
    return finish(c, mask.shape)
