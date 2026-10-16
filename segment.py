"""Find dogs in a photo and cut each one out as a clean binary mask."""
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

import env

log = logging.getLogger(__name__)

COCO_CAT, COCO_DOG = 15, 16
# lying or curled-up dogs are often labelled "cat"; these photos are assumed to be of dogs
ANIMAL_CLASSES = [COCO_CAT, COCO_DOG]


@dataclass
class Dog:
    mask: np.ndarray  # bool HxW
    box: tuple[int, int, int, int]
    score: float


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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


class DogSegmenter:
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

    def find_dogs(self, rgb: np.ndarray) -> list[Dog]:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        t = time.perf_counter()
        det = self.detector(bgr, classes=ANIMAL_CLASSES, conf=self.min_conf, agnostic_nms=True,
                            device=self.device, verbose=False)[0]
        boxes = det.boxes.xyxy.cpu().numpy().astype(int).tolist()
        scores = det.boxes.conf.cpu().numpy().tolist()
        labels = [det.names[int(c)] for c in det.boxes.cls.cpu().numpy()]
        log.info("  detect: %d dog(s) %s in %.1fs", len(boxes),
                 [f"{s:.2f}" + (" (as cat)" if lab == "cat" else "") for s, lab in zip(scores, labels)],
                 time.perf_counter() - t)
        if not boxes:
            return []

        t = time.perf_counter()
        seg = self.segmenter(bgr, bboxes=boxes, device=self.device, verbose=False)[0]
        masks = seg.masks.data.cpu().numpy() > 0.5
        log.info("  segment: %d mask(s) in %.1fs", len(masks), time.perf_counter() - t)

        dogs = [Dog(clean_mask(m), tuple(b), s) for m, b, s in zip(masks, boxes, scores)]
        return sorted(dogs, key=lambda d: d.score, reverse=True)
