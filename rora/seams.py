"""Final pass after --refine: repair what tiled VAE encoding/decoding can leave at tile edges.

To save memory, mflux encodes the drawing and decodes the result in overlapping tiles that are
blended across narrow bands. A stroke crossing a band can come out broken, and a band can leave a
straight line along it. Only pixels in and right next to the bands are changed.
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize

TILE = 512  # VAE tile side on the model canvas (px)
OVERLAP = 64  # blend band between neighbouring tiles (px)
SCALE = 8  # VAE downsampling; mflux tiles in latent units
MAX_GAP = 12  # longest break bridged (canvas px; the canvas is ~1024 px on its long side)
MARGIN = 4  # band widening, for strokes that end just outside it
MIN_SEAM_LINE = 32  # shortest straight line along a band that counts as a seam artefact
MAX_TURN = np.deg2rad(50)  # how far a bridge may bend away from both strokes' directions


def bands(length: int, tile: int = TILE, overlap: int = OVERLAP, scale: int = SCALE):
    """(start, end) of the blend bands along one side, the way mflux's VAETiler lays tiles out
    (including skipping a sliver tile at the end)."""
    n, t = length // scale, max(1, tile // scale)
    o = max(0, min(overlap // scale, t - 1))
    stride = max(1, t - o)
    out = []
    for s in range(stride, n, stride):
        if min(s + t, n) - s <= o:
            continue
        out.append((s * scale, min((s + o) * scale, length)))
    return out


def seam_bands(h: int, w: int, tile: int = TILE, overlap: int = OVERLAP):
    """(horizontal bands as y ranges, vertical bands as x ranges); none if the image fits one tile."""
    if h // SCALE <= tile // SCALE and w // SCALE <= tile // SCALE:
        return [], []
    return bands(h, tile, overlap), bands(w, tile, overlap)


def _zone(shape, rows, cols, pad):
    zone = np.zeros(shape, bool)
    for y0, y1 in rows:
        zone[max(0, y0 - pad):y1 + pad, :] = True
    for x0, x1 in cols:
        zone[:, max(0, x0 - pad):x1 + pad] = True
    return zone


def _seam_lines(ink, rows, cols):
    """Thin, perfectly straight, axis-aligned components lying inside a band."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    drop = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h = stats[i, :4]
        for x0, x1 in cols:  # vertical line along a vertical band
            if (h >= MIN_SEAM_LINE and w <= 4 and x >= x0 - MARGIN and x + w <= x1 + MARGIN):
                drop[i] = True
        for y0, y1 in rows:
            if (w >= MIN_SEAM_LINE and h <= 4 and y >= y0 - MARGIN and y + h <= y1 + MARGIN):
                drop[i] = True
    return drop[labels]


def _endpoints(skel):
    nb = cv2.filter2D(skel.astype(np.uint8), -1, np.ones((3, 3), np.float32),
                      borderType=cv2.BORDER_CONSTANT) - skel
    ys, xs = np.nonzero(skel & (nb == 1))
    return list(zip(ys.tolist(), xs.tolist()))


def _direction(skel, y, x, steps=8):
    """(unit vector pointing out of the stroke at endpoint (y, x), skeleton points walked)."""
    seen, py, px, path = {(y, x)}, y, x, [(y, x)]
    for _ in range(steps):
        nxt = [(py + dy, px + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dy or dx) and 0 <= py + dy < skel.shape[0] and 0 <= px + dx < skel.shape[1]
               and skel[py + dy, px + dx] and (py + dy, px + dx) not in seen]
        if not nxt:
            break
        py, px = nxt[0]
        seen.add((py, px))
        path.append((py, px))
    v = np.array([y - py, x - px], float)
    norm = np.linalg.norm(v)
    return (v / norm if norm else None), path


def _crosses_band(p, q, rows, cols):
    for y0, y1 in rows:
        lo, hi = sorted((p[0], q[0]))
        if hi >= y0 - MARGIN and lo < y1 + MARGIN:
            return True
    for x0, x1 in cols:
        lo, hi = sorted((p[1], q[1]))
        if hi >= x0 - MARGIN and lo < x1 + MARGIN:
            return True
    return False


def _bridges(ink, rows, cols):
    """Line segments joining stroke ends that face each other across a band."""
    zone = _zone(ink.shape, rows, cols, MARGIN + MAX_GAP)
    skel = skeletonize(ink)
    ends = [e for e in _endpoints(skel) if zone[e]]
    half_width = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    dirs, widths = {}, {}
    for e in ends:
        dirs[e], path = _direction(skel, *e)
        # stroke width a few px back from the (rounded, thinner) tip
        widths[e] = float(np.median([half_width[p] for p in path]))
    pairs = []
    for i, a in enumerate(ends):
        for b in ends[i + 1:]:
            gap = np.subtract(b, a).astype(float)
            d = np.linalg.norm(gap)
            if not (1.5 < d <= MAX_GAP) or dirs[a] is None or dirs[b] is None:
                continue
            u = gap / d
            if (np.arccos(np.clip(dirs[a] @ u, -1, 1)) > MAX_TURN
                    or np.arccos(np.clip(dirs[b] @ -u, -1, 1)) > MAX_TURN):
                continue
            if _crosses_band(a, b, rows, cols):
                pairs.append((d, a, b))
    used, out = set(), []
    for d, a, b in sorted(pairs):
        if a in used or b in used:
            continue
        used.update((a, b))
        # the distance transform overshoots by ~0.5 px a side; the thinner end sets the width
        width = max(1, round(2 * min(widths[a], widths[b]) - 1))
        out.append((a, b, width))
    return out


def repair(rgb: np.ndarray, tile: int = TILE, overlap: int = OVERLAP):
    """Fix seams in the model's RGB output (black ink on white). Returns (rgb, bridged, removed)."""
    rows, cols = seam_bands(*rgb.shape[:2], tile, overlap)
    if not rows and not cols:
        return rgb, 0, 0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ink = gray < 128
    drop = _seam_lines(ink, rows, cols)
    ink &= ~drop
    bridges = _bridges(ink, rows, cols)
    add = np.zeros(gray.shape, np.uint8)
    for (ay, ax), (by, bx), width in bridges:
        cv2.line(add, (ax, ay), (bx, by), 1, thickness=width)
    gray = gray.copy()
    gray[drop] = 255
    gray[add > 0] = 0
    removed = cv2.connectedComponents(drop.astype(np.uint8))[0] - 1
    return np.repeat(gray[..., None], 3, axis=2), len(bridges), removed
