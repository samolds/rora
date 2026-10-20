"""Pass 2: vote across pass-1 maps, keep only lines most methods agree on, and trace
them into a few smooth strokes (polylines) ready to be drawn in any line look.
"""
import logging
import time

import cv2
import numpy as np
from skimage.morphology import skeletonize

from . import cleanup

log = logging.getLogger(__name__)

WEIGHTS = {"xdog": 1.0, "canny": 1.0, "regions": 0.5, "hed": 1.0, "pidinet": 1.0,
           "lineart": 1.0, "anime": 1.0}


def disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def binarize(m: np.ndarray, mask: np.ndarray, coverage: float) -> np.ndarray:
    """Strongest lines only: at most `coverage` of the dog's pixels, as 1px skeleton."""
    vals = m[mask]
    thr = max(0.3, float(np.percentile(vals, 100 * (1 - coverage)))) if vals.size else 1.0
    return skeletonize(m > thr)


def consensus(maps: dict, mask: np.ndarray, min_votes: float, tolerance: int,
              coverage: float, edge_band: int) -> tuple[np.ndarray, np.ndarray]:
    votes = np.zeros(mask.shape, np.float32)
    k = disk(tolerance)
    for name, m in maps.items():
        lines = binarize(m, mask, coverage).astype(np.uint8)
        votes += WEIGHTS.get(name, 1.0) * cv2.dilate(lines, k)
    inner = cv2.erode(mask.astype(np.uint8), disk(edge_band)).astype(bool)
    keep = (votes >= min_votes) & inner
    return keep, votes


NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def trace(skel: np.ndarray) -> list[np.ndarray]:
    """Split a 1px skeleton into polylines that run between endpoints/junctions."""
    pts = set(zip(*np.nonzero(skel)))

    def nbrs(p):
        return [(p[0] + dy, p[1] + dx) for dy, dx in NEIGHBORS if (p[0] + dy, p[1] + dx) in pts]

    degree = {p: len(nbrs(p)) for p in pts}
    nodes = {p for p, d in degree.items() if d != 2}
    used_edges = set()
    paths = []

    def walk(start, nxt):
        path = [start, nxt]
        used_edges.add((start, nxt)); used_edges.add((nxt, start))
        prev, cur = start, nxt
        while cur not in nodes:
            step = [n for n in nbrs(cur) if n != prev and (cur, n) not in used_edges]
            if not step:
                break
            prev, cur = cur, step[0]
            used_edges.add((prev, cur)); used_edges.add((cur, prev))
            path.append(cur)
        return path

    for n in nodes:
        for m in nbrs(n):
            if (n, m) not in used_edges:
                paths.append(walk(n, m))
    # closed loops with no endpoint or junction
    for p in pts:
        for m in nbrs(p):
            if (p, m) not in used_edges:
                paths.append(walk(p, m))
    return [np.array([(x, y) for y, x in path], np.float32) for path in paths if len(path) > 1]


def path_length(p: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def rank(paths: list[np.ndarray], votes: np.ndarray, max_strokes: int,
         min_score: float) -> list[np.ndarray]:
    """Keep the strokes with the best agreement (weighted heavily, so short eye/nose loops
    beat long fur lines); min_score is relative to the best stroke."""
    if not paths:
        return paths
    h, w = votes.shape
    scores = []
    for q in paths:
        xi = np.clip(np.round(q[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(q[:, 1]).astype(int), 0, h - 1)
        scores.append(path_length(q) ** 0.5 * float(votes[yi, xi].mean()) ** 2.5)
    order = np.argsort(scores)[::-1]
    best = scores[order[0]]
    keep = [i for i in order if scores[i] >= min_score * best]
    if max_strokes is not None and max_strokes >= 0:
        keep = keep[:int(max_strokes)]
    return [paths[i] for i in keep]


def smooth(p: np.ndarray, sigma: float, closed: bool = False) -> np.ndarray:
    if len(p) < 5 or sigma <= 0:
        return p
    k = int(3 * sigma) * 2 + 1
    pad = k // 2
    if closed:
        ext = np.concatenate([p[-pad:], p, p[:pad]])
    else:
        ext = np.concatenate([np.repeat(p[:1], pad, 0), p, np.repeat(p[-1:], pad, 0)])
    g = cv2.getGaussianKernel(k, sigma)[:, 0]
    out = np.stack([np.convolve(ext[:, i], g, mode="valid") for i in range(2)], axis=1)
    if not closed:
        out[0], out[-1] = p[0], p[-1]
    return out.astype(np.float32)


def join_paths(paths: list[np.ndarray], gap: float) -> list[np.ndarray]:
    """Greedily join paths whose endpoints are within `gap`, so strokes run longer.

    Closest endpoint pairs are joined first; each path end is used at most once.
    """
    from scipy.spatial import cKDTree

    n = len(paths)
    if n < 2:
        return list(paths)
    # endpoint k belongs to path k // 2; even k = start, odd k = end
    ends = np.array([pt for p in paths for pt in (p[0], p[-1])], np.float32)
    pairs = cKDTree(ends).query_pairs(gap, output_type="ndarray")
    pairs = pairs[pairs[:, 0] // 2 != pairs[:, 1] // 2]
    dist = np.linalg.norm(ends[pairs[:, 0]] - ends[pairs[:, 1]], axis=1)

    parent = list(range(n))  # union-find over paths, to avoid closing loops

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    partner = {}
    for a, b in pairs[np.argsort(dist)]:
        if a in partner or b in partner or root(a // 2) == root(b // 2):
            continue  # endpoint already used, or joining would close a loop
        partner[a], partner[b] = b, a
        parent[root(a // 2)] = root(b // 2)

    out, seen = [], set()
    for start_path in range(n):
        if start_path in seen:
            continue
        # walk to one free end of this chain, then traverse it
        path, end = start_path, 2 * start_path
        while end in partner:
            other = partner[end]
            path, end = other // 2, other ^ 1
            if path == start_path:
                break
        chain, cur_path, entry = [], path, end
        while True:
            seen.add(cur_path)
            p = paths[cur_path]
            chain.append(p[::-1] if entry % 2 == 1 else p)
            exit_end = entry ^ 1
            if exit_end not in partner:
                break
            nxt = partner[exit_end]
            cur_path, entry = nxt // 2, nxt
        out.append(np.concatenate(chain))
    return out


def feature_blobs(rgb: np.ndarray, mask: np.ndarray, votes: np.ndarray, max_blobs: int,
                  darkness: float) -> list[np.ndarray]:
    """Dark, compact, high-agreement blobs (nose, pupils) as filled contours."""
    if max_blobs <= 0:
        return []
    size = float(np.sqrt(mask.sum()))
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    g = cv2.GaussianBlur(g, (0, 0), 0.003 * size)
    filled = np.where(mask, g, g[mask].mean()).astype(np.float32)
    local = cv2.GaussianBlur(filled, (0, 0), 0.08 * size) - g
    inner = cv2.erode(mask.astype(np.uint8), disk(max(3, int(0.02 * size)))).astype(bool)
    dark = ((local > darkness) & inner).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, disk(max(1, int(0.004 * size))))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    area_dog = mask.sum()
    near = cv2.dilate(votes, disk(max(2, int(0.008 * size))))
    scored = []
    for c in contours:
        a = cv2.contourArea(c)
        if not (area_dog * 0.0001 < a < area_dog * 0.03):
            continue
        hull = cv2.contourArea(cv2.convexHull(c))
        solidity = a / max(hull, 1)
        per = cv2.arcLength(c, True)
        circularity = 4 * np.pi * a / max(per * per, 1)
        if solidity < 0.75 or circularity < 0.35:
            continue
        m = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(m, [c], -1, 1, -1)
        agreement = float(near[m > 0].mean())
        contrast = float(local[m > 0].mean())
        scored.append((agreement * contrast * circularity, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for _, c in scored[:max_blobs]:
        out.append(smooth(c[:, 0, :].astype(np.float32), 0.002 * size, closed=True))
    return out


def silhouette_paths(mask: np.ndarray, sigma: float) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        p = c[:, 0, :].astype(np.float32)
        if len(p) > 20:
            p = smooth(p, sigma, closed=True)
            out.append(np.concatenate([p, p[:1]]))
    return out


def extract(maps: dict, mask: np.ndarray, rgb: np.ndarray, p: dict,
            exclude: np.ndarray | None = None) -> tuple[list, list, list, dict]:
    """Returns (interior strokes, silhouette strokes, filled feature blobs, debug images).

    Strokes inside `exclude` are cut away before ranking, so they don't use up the budget.
    """
    t = time.perf_counter()
    s = max(mask.shape) / 2500
    keep, votes = consensus(maps, mask, p["min_votes"], max(2, int(p["tolerance"] * s)),
                            p["coverage"], max(3, int(p["edge_band"] * s)))
    keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, disk(max(1, int(3 * s))))
    skel = skeletonize(keep > 0)

    min_len = p["min_length"] * s
    paths = [q for q in trace(skel) if path_length(q) >= min_len / 4]
    paths = join_paths(paths, gap=p["join_gap"] * s)
    paths = [smooth(q, p["smooth"] * s) for q in paths]
    paths = [q for q in paths if path_length(q) >= min_len]
    if exclude is not None:
        paths = cleanup.clip_paths(paths, ~exclude, min_len)
    paths = rank(paths, votes, p["max_strokes"], p["min_score"])
    paths = [cv2.approxPolyDP(q.reshape(-1, 1, 2), 0.6 * s, False)[:, 0, :].astype(np.float32)
             for q in paths]
    outline = silhouette_paths(mask, p["smooth"] * 2 * s)
    blobs = feature_blobs(rgb, mask, votes, int(p["max_blobs"]), p["darkness"])
    log.info("  pass2: %d stroke(s) + %d outline + %d feature blob(s) in %.1fs",
             len(paths), len(outline), len(blobs), time.perf_counter() - t)
    return paths, outline, blobs, {"votes": votes, "keep": keep, "skeleton": skel}
