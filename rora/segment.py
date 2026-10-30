"""Find the main subject of a photo (a person, cat or dog) and cut it out as a clean binary mask."""
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from . import env

log = logging.getLogger(__name__)
register_heif_opener()  # lets Pillow open .heic photos

COCO_KINDS = {0: "person", 15: "cat", 16: "dog"}
SUBJECTS = ("all", "auto", *COCO_KINDS.values())
FRAME = "frame"  # the kind of the one subject --whole-frame draws


@dataclass
class Subject:
    mask: np.ndarray  # bool HxW
    box: tuple[int, int, int, int]
    score: float
    kind: str  # "person", "cat" or "dog", or FRAME


def whole_frame(rgb: np.ndarray) -> list[Subject]:
    """The whole image as the only subject, for --whole-frame (no detection or segmentation)."""
    h, w = rgb.shape[:2]
    return [Subject(np.ones((h, w), bool), (0, 0, w, h), 1.0, FRAME)]


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def free_device_memory():
    """Return memory held by PyTorch's GPU allocator (after models are dropped) to the system."""
    import gc

    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_image(path: Path, work_size: int) -> np.ndarray:
    """Load as RGB uint8, honoring EXIF rotation, with the long side scaled to work_size."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    scale = work_size / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return np.asarray(img)


def clean_mask(mask: np.ndarray, smooth: int = 5) -> np.ndarray:
    """Keep the largest blob, fill its holes and smooth the edge."""
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (labels == largest).astype(np.uint8)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)

    if smooth > 0:
        k = smooth * 2 + 1
        blurred = cv2.GaussianBlur(filled.astype(np.float32), (k, k), 0)
        filled = (blurred > 0.5).astype(np.uint8)
    return filled.astype(bool)


def prominence(box, score: float, shape) -> float:
    """How much a detection looks like the photo's main subject: big, central and confident."""
    h, w = shape[:2]
    x0, y0, x1, y1 = box
    area = (x1 - x0) * (y1 - y0) / (w * h)
    cx, cy = (x0 + x1) / (2 * w) - 0.5, (y0 + y1) / (2 * h) - 0.5
    off_centre = np.hypot(cx, cy) / np.hypot(0.5, 0.5)  # 0 at the centre, 1 in a corner
    return score * area * (1 - 0.5 * off_centre)


class Segmenter:
    def __init__(self, device: str = "auto", detector: str = "yolo11x.pt",
                 segmenter: str = "sam2.1_b.pt", min_conf: float = 0.35):
        from ultralytics import SAM, YOLO

        env.disable_ultralytics_analytics()
        self.device = pick_device(device)
        self.min_conf = min_conf
        detector, segmenter = (str(p if Path(p).parent != Path(".") else env.MODELS / p)
                               for p in (detector, segmenter))
        log.info("Loading detector %s and segmenter %s on %s", detector, segmenter, self.device)
        t = time.perf_counter()
        self.detector = YOLO(detector)
        self.segmenter = SAM(segmenter)
        log.info("Models loaded in %.1fs", time.perf_counter() - t)

    def find_subjects(self, rgb: np.ndarray, subject: str = "all",
                      all_conf: float = 0.8) -> list[Subject]:
        """Subjects, main one first: the main one plus every other detected with at least
        `all_conf` confidence (`all`: any person, cat or dog; a kind: only that kind), or
        just the main one (`auto`)."""
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        classes = [c for c, k in COCO_KINDS.items() if subject in ("all", "auto", k)]

        t = time.perf_counter()
        det = self.detector(bgr, classes=classes, conf=self.min_conf, agnostic_nms=True,
                            device=self.device, verbose=False)[0]
        boxes = det.boxes.xyxy.cpu().numpy().astype(int).tolist()
        scores = det.boxes.conf.cpu().numpy().tolist()
        kinds = [COCO_KINDS[int(c)] for c in det.boxes.cls.cpu().numpy()]
        log.info("  detect: %s in %.1fs", ", ".join(f"{k} {s:.2f}" for k, s in zip(kinds, scores))
                 or "nothing", time.perf_counter() - t)
        if not boxes:
            return []
        order = sorted(range(len(boxes)), key=lambda j: prominence(boxes[j], scores[j], rgb.shape),
                       reverse=True)
        if subject == "auto":
            order = order[:1]
        else:
            order = order[:1] + [j for j in order[1:] if scores[j] >= all_conf]
        if len(boxes) > 1:
            log.info("  drawing: %s", ", ".join(f"{kinds[j]} {scores[j]:.2f}" for j in order))

        t = time.perf_counter()
        masks = self.segment(rgb, [boxes[j] for j in order])
        log.info("  segment: %.1fs", time.perf_counter() - t)
        return [Subject(m, tuple(boxes[j]), scores[j], kinds[j]) for m, j in zip(masks, order)]

    def segment(self, rgb: np.ndarray, boxes: list) -> list[np.ndarray]:
        """A cleaned-up mask (see clean_mask) of the thing in each box."""
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        seg = self.segmenter(bgr, bboxes=[list(b) for b in boxes], device=self.device,
                             verbose=False)[0]
        return [clean_mask(m) for m in seg.masks.data.cpu().numpy() > 0.5]

    def find_subject(self, rgb: np.ndarray, subject: str = "auto") -> Subject | None:
        """The main subject (or the main one of kind `subject`), or None if there is none."""
        found = self.find_subjects(rgb, "auto" if subject == "all" else subject)
        return found[0] if found else None


def describe(kinds: list[str]) -> str:
    """Plain-English list of subjects, e.g. 'a dog', 'two people and a cat'."""
    if FRAME in kinds:
        return "the whole frame"
    counts = {k: kinds.count(k) for k in dict.fromkeys(kinds)}
    plural = {"person": "people", "cat": "cats", "dog": "dogs"}
    words = {1: "a", 2: "two", 3: "three", 4: "four", 5: "five"}
    parts = [f"{words.get(n, n)} {k if n == 1 else plural[k]}" for k, n in counts.items()]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
