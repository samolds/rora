"""Draw the face features on purpose: eyes (their own traced outline, or an optional almond
template; pupils optional), a shaped nose with nostrils, and the mouth as a lip line.
Shapes come from each feature's SAM mask, then get tidied like an artist would.

Returns strokes (polylines drawn in the current line look) and fills. A fill is a list of
contours: the first is filled with ink, the rest are cut out of it (e.g. the eye highlight).
"""
import cv2
import numpy as np

import cleanup
from features import Face, Feature


def rot(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], np.float32)


def ellipse(center, rx, ry, angle, n=48) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([rx * np.cos(t), ry * np.sin(t)], 1) @ rot(angle).T
    return (pts + center).astype(np.float32)


def close(p: np.ndarray) -> np.ndarray:
    return np.concatenate([p, p[:1]])


def mask_points(f: Feature) -> np.ndarray:
    ys, xs = np.nonzero(f.mask)
    return np.stack([xs, ys], 1).astype(np.float32)


def has_mask(f: Feature | None) -> bool:
    return f is not None and f.mask is not None and f.mask.sum() > 20


def face_angle(face: Face) -> float:
    """Tilt of the face, from the line through the eyes (0 if unknown)."""
    if len(face.eyes) == 2:
        d = face.eyes[1].center - face.eyes[0].center
        return float(np.arctan2(d[1], d[0]))
    return 0.0


def wrap_half(a: float) -> float:
    """Wrap an axis angle to (-pi/2, pi/2]; an axis has no direction."""
    return (a + np.pi / 2) % np.pi - np.pi / 2


def facing_us(face: Face, mask: np.ndarray) -> bool:
    """Frontal view: two eyes, or (eyes hidden) the nose sits mid-way across the head's
    silhouette rather than at its edge, as it would in profile."""
    if len(face.eyes) == 2:
        return True
    if face.nose is None or face.eyes:
        return False
    x0, _, x1, _ = map(int, face.head.box)
    row = mask[int(face.nose.center[1]), max(0, x0):x1]
    xs = np.flatnonzero(row)
    if len(xs) < 2:
        return False
    left, right = x0 + xs[0], x0 + xs[-1]
    pos = (face.nose.center[0] - left) / max(right - left, 1)
    return 0.3 < pos < 0.7


# ---------- eyes ----------

def eye_shape(f: Feature):
    """Center, half-length, half-height and axis angle from the eye mask (PCA)."""
    pts = mask_points(f)
    c = pts.mean(0)
    vals, vecs = np.linalg.eigh(np.cov((pts - c).T))
    major = vecs[:, 1]
    a, b = 2 * np.sqrt(max(vals[1], 1e-6)), 2 * np.sqrt(max(vals[0], 1e-6))
    return c, a, b, float(np.arctan2(major[1], major[0]))


def almond(c, a, b, angle, n=40) -> np.ndarray:
    t = np.linspace(-1, 1, n)
    upper = np.stack([a * t, -b * (1 - t ** 2) ** 0.8], 1)
    lower = np.stack([a * t[::-1], b * (1 - t[::-1] ** 2) ** 0.95], 1)
    pts = np.concatenate([upper, lower[1:-1]]) @ rot(angle).T
    return (pts + c).astype(np.float32)


def darkest_point(rgb, f: Feature, c) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ys, xs = np.nonzero(f.mask)
    if len(xs) < 5:
        return c
    g = cv2.GaussianBlur(g, (0, 0), max(1.0, 0.1 * np.sqrt(len(xs))))
    vals = g[ys, xs]
    dark = vals <= np.percentile(vals, 10)
    return np.array([xs[dark].mean(), ys[dark].mean()], np.float32)


def traced_outline(f: Feature, smooth: float, step: float) -> np.ndarray:
    """The feature's own outline, lightly smoothed (no template shape imposed)."""
    cnts, _ = cv2.findContours(f.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float32)
    return cleanup.spline(c, tol=smooth, step=step, closed=True)


def place_pupil(outline: np.ndarray, want: np.ndarray, center: np.ndarray, r: float):
    """Move the pupil toward the eye's center until it fits inside the outline; shrink if needed."""
    poly = outline.reshape(-1, 1, 2)
    for t in np.linspace(0, 1, 11):
        p = (1 - t) * want + t * center
        room = cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), True)
        if room >= 0.9 * r:
            return p, r
    room = cv2.pointPolygonTest(poly, (float(center[0]), float(center[1])), True)
    return center, max(1.0, 0.85 * room)


def draw_eyes(face: Face, rgb, size: float, almond_eyes: bool,
              pupils: bool) -> tuple[list, list]:
    eyes = [e for e in face.eyes if has_mask(e)]
    if not eyes:
        return [], []
    shapes = [eye_shape(e) for e in eyes]
    base = face_angle(face)
    step = max(1.0, 0.002 * size)

    if almond_eyes:
        # same size for both eyes; almond aspect kept in a pleasing range
        a = float(np.mean([s[1] for s in shapes]))
        b = float(np.clip(np.mean([s[2] for s in shapes]), 0.45 * a, 0.75 * a))
        if len(shapes) == 2:
            rel = [wrap_half(s[3] - base) for s in shapes]
            tilt = float(np.clip((rel[0] - rel[1]) / 2, -np.radians(25), np.radians(25)))
            angles = [base + tilt, base - tilt]  # mirror-symmetric
        else:
            angles = [base + float(np.clip(wrap_half(shapes[0][3] - base),
                                           -np.radians(30), np.radians(30)))]
        outlines = [almond(s[0], a, b, ang) for s, ang in zip(shapes, angles)]
    else:
        outlines = [traced_outline(e, smooth=0.05 * e.size, step=step) for e in eyes]

    strokes, fills = [], []
    for e, (c, _, b_own, _), outline in zip(eyes, shapes, outlines):
        strokes.append(close(outline) if almond_eyes else outline)
        if not pupils:
            continue
        r = 0.62 * (b if almond_eyes else b_own)
        pupil_c, r = place_pupil(outline, darkest_point(rgb, e, c), c, r)
        hl = pupil_c + np.array([-0.35 * r, -0.35 * r], np.float32) @ rot(base).T
        fills.append([ellipse(pupil_c, r, r, 0), ellipse(hl, 0.32 * r, 0.32 * r, 0, 24)])
    return strokes, fills


# ---------- nose ----------

def to_face(pts, origin, angle):
    return (pts - origin) @ rot(angle)


def from_face(pts, origin, angle):
    return (pts @ rot(angle).T + origin).astype(np.float32)


def nose_outline(f: Feature, size: float) -> np.ndarray:
    cnts, _ = cv2.findContours(f.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = close(cv2.convexHull(max(cnts, key=cv2.contourArea))[:, 0, :].astype(np.float32))
    # resample the hull's straight edges densely so the spline can round them off
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(hull, axis=0), axis=1))]
    t = np.arange(0, d[-1], max(1.0, 0.002 * size))
    dense = np.stack([np.interp(t, d, hull[:, 0]), np.interp(t, d, hull[:, 1])], 1)
    return cleanup.spline(dense.astype(np.float32), tol=0.004 * size,
                          step=max(1.0, 0.002 * size), closed=True)


def draw_nose(face: Face, size: float, solid: bool,
              frontal: bool) -> tuple[list, list, np.ndarray | None]:
    """Returns strokes, fills, and the nose bottom-center point (for the philtrum)."""
    f = face.nose
    if not has_mask(f):
        return [], [], None
    ang = face_angle(face)
    outline = nose_outline(f, size)
    c = outline.mean(0)
    local = to_face(outline, c, ang)
    (x0, y0), (x1, y1) = local.min(0), local.max(0)
    w, h = x1 - x0, y1 - y0
    bottom = from_face(np.array([[0, y1]], np.float32), c, ang)[0]

    if solid:
        hl = from_face(np.array([[-0.2 * w, y0 + 0.28 * h]], np.float32), c, ang)[0]
        return [], [[outline, ellipse(hl, 0.14 * w, 0.08 * h + 1, ang, 24)]], bottom

    if frontal:  # two nostrils
        spots = [(-0.26 * w, 0.18 * h, -0.35), (0.26 * w, 0.18 * h, 0.35)]
    else:
        spots = [(0.0, 0.18 * h, 0.0)]
    fills = []
    for dx, dy, tilt in spots:
        p = from_face(np.array([[dx, dy]], np.float32), c, ang)[0]
        fills.append([ellipse(p, 0.11 * w, 0.07 * w, ang + tilt, 24)])
    return [outline], fills, bottom


# ---------- mouth ----------

def edge_profile(pts_local: np.ndarray, top: bool, bins: int) -> np.ndarray:
    """Upper (or lower) edge of a point cloud in face coordinates, as a polyline."""
    x = pts_local[:, 0]
    edges = np.linspace(x.min(), x.max(), bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
    out = []
    for i in range(bins):
        ys = pts_local[idx == i, 1]
        if len(ys):
            out.append(((edges[i] + edges[i + 1]) / 2, ys.min() if top else ys.max()))
    return np.array(out, np.float32)


def draw_mouth(face: Face, size: float, nose_bottom, frontal: bool) -> list:
    f = face.mouth
    if not has_mask(f):
        return []
    ang = face_angle(face)
    pts = mask_points(f)
    c = pts.mean(0)
    local = to_face(pts, c, ang)
    w = float(np.ptp(local[:, 0]))
    h = float(np.ptp(local[:, 1]))
    bins = max(8, int(w / max(2.0, 0.006 * size)))
    step = max(1.0, 0.002 * size)
    tol = 0.004 * size

    upper = edge_profile(local, top=True, bins=bins)
    lines = [cleanup.spline(from_face(upper, c, ang), tol, step)]
    if h > 0.45 * w:  # open mouth: also draw the jaw line
        lower = edge_profile(local, top=False, bins=bins)
        lines.append(cleanup.spline(from_face(lower, c, ang), tol, step))

    if nose_bottom is not None and frontal:
        # philtrum: short line from the nose down to the lip line
        lip = lines[0]
        target = lip[np.argmin(np.linalg.norm(lip - nose_bottom, axis=1))]
        gap = float(np.linalg.norm(target - nose_bottom))
        if 0 < gap < 0.25 * size:
            lines.append(np.linspace(nose_bottom, target, 12).astype(np.float32))
    return lines


# ---------- entry point ----------

def draw_face(face: Face, rgb: np.ndarray, mask: np.ndarray, size: float, nose_solid: bool,
              almond_eyes: bool = False, pupils: bool = False) -> tuple[list, list]:
    frontal = facing_us(face, mask)
    eye_strokes, eye_fills = draw_eyes(face, rgb, size, almond_eyes, pupils)
    nose_strokes, nose_fills, nose_bottom = draw_nose(face, size, nose_solid, frontal)
    mouth_strokes = draw_mouth(face, size, nose_bottom, frontal)
    return eye_strokes + nose_strokes + mouth_strokes, eye_fills + nose_fills
