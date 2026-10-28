"""Keep subjects steady through an animation: a subject the detector misses in some frames (an
odd pose, motion blur) is segmented again there from where it is in the frames around them, so
it doesn't blink in and out."""
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .segment import Subject

# a re-segmented mask this many times bigger or smaller than the subject's nearest detected
# mask is probably something else (the subject really left, or the box landed on background)
MAX_AREA_CHANGE = 4.0


@dataclass
class Track:
    """One subject followed through the frames it was detected in."""
    kind: str
    seen: dict[int, Subject] = field(default_factory=dict)  # frame index -> detection

    @property
    def last(self) -> tuple[int, Subject]:
        i = max(self.seen)
        return i, self.seen[i]


def iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _distance(a, b) -> float:
    """Centre distance in units of the bigger box's diagonal."""
    diag = max(np.hypot(a[2] - a[0], a[3] - a[1]), np.hypot(b[2] - b[0], b[3] - b[1]), 1)
    return np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2) / diag


def link(frames: list[list[Subject]]) -> list[Track]:
    """Follow each subject through the frames: a detection joins the track of the same kind
    whose latest box overlaps it or is close by (closer the fewer frames ago it was seen);
    otherwise it starts a new track."""
    tracks: list[Track] = []
    for i, subjects in enumerate(frames):
        pairs = []
        for ti, t in enumerate(tracks):
            j, prev = t.last
            for si, s in enumerate(subjects):
                if s.kind != t.kind:
                    continue
                overlap, dist = iou(prev.box, s.box), _distance(prev.box, s.box)
                if overlap >= 0.1 or dist < 0.5 * (i - j):
                    pairs.append((overlap - dist, ti, si))
        used_t, used_s = set(), set()
        for _, ti, si in sorted(pairs, reverse=True):
            if ti not in used_t and si not in used_s:
                tracks[ti].seen[i] = subjects[si]
                used_t.add(ti)
                used_s.add(si)
        tracks += [Track(s.kind, {i: s}) for si, s in enumerate(subjects) if si not in used_s]
    return tracks


def _box_at(track: Track, i: int) -> tuple[int, int, int, int]:
    """The track's box in frame i, interpolated between the detections either side of it."""
    before = [j for j in track.seen if j < i]
    after = [j for j in track.seen if j > i]
    if not before or not after:
        return track.seen[max(before) if before else min(after)].box
    j0, j1 = max(before), min(after)
    a, b = np.array(track.seen[j0].box, float), np.array(track.seen[j1].box, float)
    return tuple(np.round(a + (b - a) * (i - j0) / (j1 - j0)).astype(int).tolist())


def _nearest(track: Track, i: int) -> Subject:
    return track.seen[min(track.seen, key=lambda j: (abs(j - i), j))]


def steady(frames: list[list[Subject]],
           segment: Callable[[int, list], list[np.ndarray]],
           min_share: float = 0.5) -> tuple[list[list[Subject]], dict[str, list[int]]]:
    """Fill in subjects detected in more than `min_share` of the frames wherever they were
    missed. `segment(i, boxes)` cuts out one mask per box in frame i. Returns the frames'
    subjects (the filled-in ones with score 0) and, per subject, the frames filled in."""
    n = len(frames)
    tracks = [t for t in link(frames) if len(t.seen) < n and len(t.seen) > min_share * n]
    wanted: dict[int, list[Track]] = {}
    for t in tracks:
        for i in range(n):
            if i not in t.seen:
                wanted.setdefault(i, []).append(t)

    kinds = [t.kind for t in tracks]
    labels = {id(t): t.kind if kinds.count(t.kind) == 1
              else f"{t.kind}{kinds[:k + 1].count(t.kind)}" for k, t in enumerate(tracks)}
    out = [list(subjects) for subjects in frames]
    filled: dict[str, list[int]] = {}
    for i, missing in sorted(wanted.items()):
        boxes = [_box_at(t, i) for t in missing]
        for t, box, mask in zip(missing, boxes, segment(i, boxes)):
            area, ref = mask.sum(), _nearest(t, i).mask.sum()
            if not area or not 1 / MAX_AREA_CHANGE <= area / max(ref, 1) <= MAX_AREA_CHANGE:
                continue
            taken = np.logical_or.reduce([s.mask for s in out[i]]) if out[i] else None
            if taken is not None and (mask & taken).sum() > 0.5 * area:
                continue  # landed on a subject that is already drawn
            out[i].append(Subject(mask, box, 0.0, t.kind))
            filled.setdefault(labels[id(t)], []).append(i)
    return out, filled
