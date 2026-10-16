"""Find the dog's head and its features (eyes, nose, mouth) with OWLv2, zooming into the head."""
import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

import cleanup
import edgemaps
import env
import strokes

log = logging.getLogger(__name__)

OWL_MODEL = "google/owlv2-base-patch16-ensemble"
HEAD_QUERIES = ["a dog's head", "a dog's face"]
FEATURE_QUERIES = {
    "eye": ["a dog's eye", "an eye"],
    "nose": ["a dog's nose", "a black nose"],
    "mouth": ["a dog's mouth", "an open mouth with tongue"],
}


@dataclass
class Feature:
    name: str
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in full-image pixels
    score: float
    mask: np.ndarray | None = None

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])

    @property
    def size(self) -> float:
        return float(np.sqrt((self.box[2] - self.box[0]) * (self.box[3] - self.box[1])))


@dataclass
class Face:
    head: Feature
    eyes: list[Feature] = field(default_factory=list)
    nose: Feature | None = None
    mouth: Feature | None = None
    candidates: list[Feature] = field(default_factory=list)

    def all(self) -> list[Feature]:
        return self.eyes + [f for f in (self.nose, self.mouth) if f is not None]


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1e-6)


def nms(dets: list[Feature], iou: float = 0.4) -> list[Feature]:
    out = []
    for d in sorted(dets, key=lambda d: d.score, reverse=True):
        if all(box_iou(d.box, o.box) < iou for o in out):
            out.append(d)
    return out


class FeatureFinder:
    def __init__(self, device: str = "cpu", min_score: float = 0.1):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        t = time.perf_counter()
        self.processor = Owlv2Processor.from_pretrained(OWL_MODEL)
        self.model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL).eval()
        self.device = device
        try:
            self.model.to(device)
        except Exception as e:  # fall back if the GPU backend lacks an op
            log.info("  OWLv2 stays on cpu (%s)", e)
            self.device = "cpu"
        self.torch = torch
        self.min_score = min_score
        log.info("  loaded OWLv2 in %.1fs on %s", time.perf_counter() - t, self.device)
        self._sam = None

    @property
    def sam(self):
        if self._sam is None:
            from ultralytics import SAM

            env.disable_ultralytics_analytics()
            self._sam = SAM(str(env.MODELS / "sam2.1_b.pt"))
        return self._sam

    def detect(self, rgb: np.ndarray, queries: dict[str, list[str]]) -> list[Feature]:
        """Run OWLv2 on an RGB crop; returns boxes (crop pixels) labelled with the query group."""
        labels = [(name, q) for name, qs in queries.items() for q in qs]
        img = Image.fromarray(rgb)
        inputs = self.processor(text=[[q for _, q in labels]], images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model(**inputs)
        # OWLv2 pads to a square, so boxes are relative to the padded size
        side = max(img.size)
        res = self.processor.post_process_grounded_object_detection(
            out, threshold=self.min_score, target_sizes=[(side, side)])[0]
        dets = []
        for box, score, li in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(),
                                  res["labels"].cpu().numpy()):
            x0, y0, x1, y1 = np.clip(box, 0, [img.width, img.height, img.width, img.height])
            if x1 - x0 > 2 and y1 - y0 > 2:
                dets.append(Feature(labels[int(li)][0], (x0, y0, x1, y1), float(score)))
        return dets

    def find_head(self, rgb: np.ndarray, mask: np.ndarray) -> Feature | None:
        ys, xs = np.nonzero(mask)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        crop = rgb[y0:y1, x0:x1].copy()
        crop[~mask[y0:y1, x0:x1]] = 255
        dets = self.detect(crop, {"head": HEAD_QUERIES})
        if not dets:
            return None
        best = nms(dets)[0]
        b = best.box
        return Feature("head", (b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0), best.score)

    def find(self, rgb: np.ndarray, mask: np.ndarray, pad: float = 0.15) -> Face | None:
        t = time.perf_counter()
        head = self.find_head(rgb, mask)
        if head is None:
            log.info("  features: no head found")
            return None
        h, w = mask.shape
        bx0, by0, bx1, by1 = head.box
        px, py = (bx1 - bx0) * pad, (by1 - by0) * pad
        x0, y0 = int(max(0, bx0 - px)), int(max(0, by0 - py))
        x1, y1 = int(min(w, bx1 + px)), int(min(h, by1 + py))
        dets = self.detect(rgb[y0:y1, x0:x1], FEATURE_QUERIES)
        for d in dets:
            d.box = (d.box[0] + x0, d.box[1] + y0, d.box[2] + x0, d.box[3] + y0)
        face = Face(head=head, candidates=dets)
        log.info("  features: head %.2f, %d candidate(s) in %.1fs", head.score, len(dets),
                 time.perf_counter() - t)
        return face

    def find_features(self, rgb: np.ndarray, mask: np.ndarray) -> Face | None:
        """Head + the most plausible eyes/nose/mouth, each with a SAM 2 shape mask."""
        face = self.find(rgb, mask)
        if face is None:
            return None
        select(face)
        add_masks(face, rgb, mask, self.sam, self.device)
        log.info("  features: %d eye(s), nose %s, mouth %s", len(face.eyes),
                 "yes" if face.nose else "no", "yes" if face.mouth else "no")
        return face


def inside(p: np.ndarray, box) -> bool:
    return box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


def select(face: Face, min_eye: float = 0.2, min_nose: float = 0.2, min_mouth: float = 0.15):
    """Pick a plausible nose, up to two eyes, and a mouth from the raw candidates."""
    by = {n: nms([c for c in face.candidates if c.name == n]) for n in FEATURE_QUERIES}

    noses = [n for n in by["nose"] if n.score >= min_nose]
    face.nose = noses[0] if noses else None

    def clear_of_nose(f: Feature) -> bool:
        return face.nose is None or (box_iou(f.box, face.nose.box) < 0.1
                                     and not inside(f.center, face.nose.box))

    eyes = [e for e in by["eye"] if e.score >= min_eye and clear_of_nose(e)]
    if eyes:
        first = eyes[0]
        face.eyes = [first]
        for e in eyes[1:]:
            similar = 0.5 < e.size / first.size < 2 and e.score >= 0.5 * first.score
            apart = np.linalg.norm(e.center - first.center) > 1.2 * first.size
            if similar and apart:
                face.eyes.append(e)
                break
        face.eyes.sort(key=lambda e: e.center[0])

    mouths = [m for m in by["mouth"] if m.score >= min_mouth and clear_of_nose(m)
              and not any(inside(m.center, e.box) for e in face.eyes)]
    face.mouth = mouths[0] if mouths else None


def add_masks(face: Face, rgb: np.ndarray, dog_mask: np.ndarray, sam, device: str):
    """Trace each feature's exact shape with SAM 2 on the head crop (more pixels per feature)."""
    feats = face.all()
    if not feats:
        return
    h, w = dog_mask.shape
    bx0, by0, bx1, by1 = face.head.box
    pad = 0.1 * max(bx1 - bx0, by1 - by0)
    x0, y0 = int(max(0, bx0 - pad)), int(max(0, by0 - pad))
    x1, y1 = int(min(w, bx1 + pad)), int(min(h, by1 + pad))
    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1][..., ::-1])  # ultralytics expects BGR
    boxes = [[f.box[0] - x0, f.box[1] - y0, f.box[2] - x0, f.box[3] - y0] for f in feats]
    res = sam(crop, bboxes=boxes, device=device, verbose=False)[0]
    masks = res.masks.data.cpu().numpy() > 0.5
    for f, m in zip(feats, masks):
        full = np.zeros((h, w), bool)
        full[y0:y1, x0:x1] = m
        # SAM can leak into fur; keep it within a slightly enlarged detection box
        fx0, fy0, fx1, fy1 = f.box
        gx, gy = 0.15 * (fx1 - fx0), 0.15 * (fy1 - fy0)
        limit = np.zeros((h, w), bool)
        limit[int(max(0, fy0 - gy)):int(fy1 + gy), int(max(0, fx0 - gx)):int(fx1 + gx)] = True
        f.mask = full & limit & dog_mask


# ---------- head pass: redraw the head zoomed-in, keep features clear ----------

def head_window(face: Face, shape, pad: float = 0.08) -> tuple[int, int, int, int]:
    h, w = shape
    x0, y0, x1, y1 = face.head.box
    p = pad * max(x1 - x0, y1 - y0)
    return int(max(0, x0 - p)), int(max(0, y0 - p)), int(min(w, x1 + p)), int(min(h, y1 + p))


ZONE_GROW = {"eye": 0.6, "nose": 0.3, "mouth": 0.1}  # clear margin, relative to feature size


def feature_zone(face: Face, shape, grow: float) -> np.ndarray:
    """Union of feature shapes, each grown by `grow` px plus a margin relative to its size;
    other lines must stay out of it (the features are drawn separately)."""
    zone = np.zeros(shape, bool)
    for f in face.all():
        m = np.zeros(shape, np.uint8)
        if f.mask is not None and f.mask.any():
            m[f.mask] = 1
        else:
            x0, y0, x1, y1 = map(int, f.box)
            m[y0:y1, x0:x1] = 1
        r = int(grow + ZONE_GROW.get(f.name, 0) * f.size)
        if r >= 1:
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        zone |= m.astype(bool)
    return zone


def head_maps(rgb: np.ndarray, mask: np.ndarray, face: Face, device: str):
    """Pass 1 on the head crop alone (many more pixels per feature). Returns None if too small."""
    x0, y0, x1, y1 = head_window(face, mask.shape)
    crop_rgb, crop_mask = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    if crop_mask.sum() < 1000:
        return None
    return edgemaps.candidate_maps(crop_rgb, crop_mask, device), crop_rgb, crop_mask, (x0, y0)


def head_strokes(hm, p: dict) -> list[np.ndarray]:
    """Pass 2 on the head crop, in full-image coordinates."""
    if hm is None:
        return []
    maps, crop_rgb, crop_mask, (x0, y0) = hm
    inner, _, _, _ = strokes.extract(maps, crop_mask, crop_rgb, p)
    return [q + np.array([x0, y0], np.float32) for q in inner]


def feature_outlines(face: Face, size: float) -> list[np.ndarray]:
    """Smoothed outline of each feature shape (interim drawing until features get styled)."""
    out = []
    for f in face.all():
        if f.mask is None or not f.mask.any():
            continue
        cnts, _ = cv2.findContours(f.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        if len(c) >= 8:
            out.append(cleanup.spline(c[:, 0, :].astype(np.float32), tol=0.002 * size,
                                      step=max(1.0, 0.002 * size), closed=True))
    return out
