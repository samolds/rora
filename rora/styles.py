"""Line-art styles. Each style turns (rgb, dog mask) into a boolean "ink" mask.

Tonal styles put ink where the photo is light when printing light ink (and the reverse
for dark ink), so the result never looks like a photo negative.
"""
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np


@dataclass
class Style:
    fn: Callable
    description: str
    defaults: dict = field(default_factory=dict)
    clip: bool = True


STYLES: dict[str, Style] = {}


def register(name: str, description: str, **defaults):
    def wrap(fn):
        STYLES[name] = Style(fn, description, {"detail": 0.5, "weight": 2, **defaults})
        return fn
    return wrap


# ---------- helpers ----------

def gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def tone(rgb: np.ndarray, mask: np.ndarray, contrast: float = 2.0) -> np.ndarray:
    """Local-contrast-boosted brightness in [0,1], stretched to the dog's own range."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if contrast > 0:
        g = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8, 8)).apply(g)
    g = g.astype(np.float32) / 255.0
    lo, hi = np.percentile(g[mask], [2, 98]) if mask.any() else (0.0, 1.0)
    return np.clip((g - lo) / max(hi - lo, 1e-3), 0, 1)


def density(rgb, mask, p) -> np.ndarray:
    """How much ink each pixel should get, 0..1, respecting ink colour and --invert."""
    t = tone(rgb, mask, p.get("contrast", 2.0))
    return t if p["ink_light"] else 1.0 - t


def thicken(ink: np.ndarray, weight: int) -> np.ndarray:
    if weight <= 1:
        return ink
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (weight, weight))
    return cv2.dilate(ink.astype(np.uint8), k).astype(bool)


def silhouette(mask: np.ndarray, width: int) -> np.ndarray:
    m = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width * 2 + 1, width * 2 + 1))
    return (m - cv2.erode(m, k)).astype(bool)


def scale(rgb: np.ndarray) -> float:
    """Size factor so pixel-based settings behave the same at any working size."""
    return max(rgb.shape[:2]) / 2500.0


# ---------- original line styles ----------

@register("stencil", "Bold two-tone fur texture (like 1.png / rora.png)",
          threshold=0.5, contrast=3.0)
def stencil(rgb, mask, p):
    d = density(rgb, mask, p)
    sigma = (0.5 + (1 - p["detail"]) * 4) * scale(rgb)
    d = cv2.GaussianBlur(d, (0, 0), sigma)
    return d > p["threshold"]


@register("ink", "Clean pen-and-ink lines (XDoG)", epsilon=0.02)
def ink(rgb, mask, p):
    g = gray(rgb)
    s = (0.6 + (1 - p["detail"]) * 1.6) * scale(rgb)
    g1 = cv2.GaussianBlur(g, (0, 0), s)
    g2 = cv2.GaussianBlur(g, (0, 0), s * 1.6)
    lines = (g1 - 0.98 * g2) < -p["epsilon"]
    return thicken(lines, p["weight"] - 1)


@register("edges", "Fine, sketchy edge lines (like sobel.png)", blur=1.0)
def edges(rgb, mask, p):
    g = cv2.GaussianBlur(gray(rgb), (0, 0), p["blur"] * scale(rgb))
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    cutoff = np.percentile(mag[mask], 92 - p["detail"] * 20)
    return thicken(mag > cutoff, p["weight"] - 1)


@register("outline", "Minimal: silhouette plus main features (eyes, nose, mouth)",
          weight=4, min_length=120)
def outline(rgb, mask, p):
    s = scale(rgb)
    smooth = cv2.bilateralFilter(rgb, 9, 60, 9)
    g = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 2 * s)
    lo = int(40 + (1 - p["detail"]) * 60)
    e = cv2.Canny(g, lo, lo * 2) > 0
    e &= cv2.erode(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(e.astype(np.uint8), connectivity=8)
    min_len = p["min_length"] * (1.2 - p["detail"]) * s
    keep = np.zeros(n, bool)
    keep[1:] = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]) >= min_len
    features = thicken(keep[labels], p["weight"] - 1)
    return features | silhouette(mask, p["weight"])


# ---------- photo-real tonal styles ----------

def rotated_coords(shape, angle_deg: float):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    a = np.deg2rad(angle_deg)
    return xx * np.cos(a) + yy * np.sin(a), -xx * np.sin(a) + yy * np.cos(a)


@register("halftone", "Photo-real newspaper dot screen", cell=14, angle=45, contrast=2.0)
def halftone(rgb, mask, p):
    cell = max(3.0, p["cell"] * (1.5 - p["detail"]) * scale(rgb))
    d = cv2.GaussianBlur(density(rgb, mask, p), (0, 0), cell / 3)
    u, v = rotated_coords(d.shape, p["angle"])
    fu = u / cell - np.floor(u / cell) - 0.5
    fv = v / cell - np.floor(v / cell) - 0.5
    return np.sqrt(fu ** 2 + fv ** 2) < np.sqrt(d / np.pi)


@register("stipple", "Photo-real hand-stippled dots", spacing=8, dot=5, seed=0, contrast=2.0)
def stipple(rgb, mask, p):
    s = scale(rgb)
    step = max(4, int(round(p["spacing"] * (1.5 - p["detail"]) * s)))
    r = max(2, int(round(p["dot"] * s / 2)))  # radius 2+ so --min-speck doesn't erase dots
    d = density(rgb, mask, p)
    h, w = d.shape
    small = cv2.resize(d, (w // step, h // step), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(int(p["seed"]))
    ys, xs = np.nonzero(small > rng.random(small.shape))
    jitter = rng.uniform(-0.4, 0.4, (2, len(xs))) * step
    out = np.zeros((h, w), np.uint8)
    for x, y in zip(((xs + 0.5) * step + jitter[0]).astype(int),
                    ((ys + 0.5) * step + jitter[1]).astype(int)):
        cv2.circle(out, (int(x), int(y)), r, 1, -1)
    return out.astype(bool)


@register("engraving", "Photo-real engraved line shading (banknote look)",
          spacing=10, angle=30, crosshatch=0.65, contrast=2.0)
def engraving(rgb, mask, p):
    spacing = max(3.0, p["spacing"] * (1.5 - p["detail"]) * scale(rgb))
    d = cv2.GaussianBlur(density(rgb, mask, p), (0, 0), spacing / 4)

    def lines(angle, amount):
        _, v = rotated_coords(d.shape, angle)
        wobble = d * spacing * 0.5  # bend lines with the form, like a burin following fur
        frac = ((v + wobble) / spacing) % 1.0
        return np.abs(frac - 0.5) < amount / 2

    ink = lines(p["angle"], d)
    dense = np.clip((d - p["crosshatch"]) / max(1 - p["crosshatch"], 1e-3), 0, 1)
    return ink | (lines(p["angle"] + 90, dense) & (dense > 0))


# ---------- cartoony styles ----------

def posterize(rgb, mask, p, levels: int) -> np.ndarray:
    """Flat tone bands 0..levels-1 after edge-preserving smoothing."""
    smooth = rgb
    for _ in range(3):
        smooth = cv2.bilateralFilter(smooth, 9, 50, 7)
    d = density(smooth, mask, p)
    d = cv2.medianBlur((d * 255).astype(np.uint8), 7).astype(np.float32) / 255
    return np.minimum((d * levels).astype(int), levels - 1)


def band_edges(q: np.ndarray, weight: int) -> np.ndarray:
    e = cv2.Canny((q * (255 // max(q.max(), 1))).astype(np.uint8), 1, 1) > 0
    return thicken(e, weight)


@register("cartoon", "Bold outlines and flat highlight shapes", weight=5, levels=3)
def cartoon(rgb, mask, p):
    s = scale(rgb)
    w = max(2, int(p["weight"] * s))
    q = posterize(rgb, mask, p, int(p["levels"]))
    fill = q == q.max()
    lines = band_edges(q, max(1, w // 2)) if p["detail"] > 0.3 else np.zeros_like(fill)
    return fill | lines | silhouette(mask, w)


@register("comic", "Comic-book: outlines, solid highlights, halftone mid-tones",
          weight=4, levels=3, cell=12, angle=45)
def comic(rgb, mask, p):
    s = scale(rgb)
    w = max(2, int(p["weight"] * s))
    q = posterize(rgb, mask, p, int(p["levels"]))
    dots = halftone(rgb, mask, {**p, "contrast": 0})
    mid = (q > 0) & (q < q.max())
    return (q == q.max()) | (mid & dots) | band_edges(q, max(1, w // 2)) | silhouette(mask, w)


# ---------- dramatic / epic styles ----------

@register("woodcut", "Dramatic carved relief: solid lights, gouged mid-tones",
          spacing=16, angle=-20, light=0.7, contrast=3.5)
def woodcut(rgb, mask, p):
    spacing = max(4.0, p["spacing"] * (1.5 - p["detail"]) * scale(rgb))
    d = cv2.GaussianBlur(density(rgb, mask, p), (0, 0), spacing / 3)
    _, v = rotated_coords(d.shape, p["angle"])
    frac = ((v + np.sin(v / (spacing * 6)) * spacing) / spacing) % 1.0
    gouges = np.abs(frac - 0.5) < d * 0.6
    return (d > p["light"]) | (gouges & (d > 0.15))


@register("noir", "Chiaroscuro: only the brightest highlights plus a hard rim light",
          light=0.62, rim=10, light_angle=-45, contrast=4.0)
def noir(rgb, mask, p):
    s = scale(rgb)
    d = cv2.GaussianBlur(density(rgb, mask, p), (0, 0), 2 * s)
    threshold = p["light"] + (0.5 - p["detail"]) * 0.2
    a = np.deg2rad(p["light_angle"])
    shift = p["rim"] * s
    m = mask.astype(np.uint8)
    # with --whole-frame the mask is the frame, which has no edge to light (no rim along the border)
    border = cv2.BORDER_REPLICATE if mask.all() else cv2.BORDER_CONSTANT
    moved = cv2.warpAffine(m, np.float32([[1, 0, -np.cos(a) * shift],
                                          [0, 1, -np.sin(a) * shift]]), m.shape[::-1],
                           borderMode=border)
    rim = mask & ~moved.astype(bool)
    return (d > threshold) | rim


@register("sketch", "Loose pencil sketch with grainy shading", weight=1, grain=0.35)
def sketch(rgb, mask, p):
    s = scale(rgb)
    g = gray(rgb)
    blur = cv2.GaussianBlur(1 - g, (0, 0), (4 + p["detail"] * 6) * s)
    dodge = np.clip(g / np.maximum(1 - blur, 1e-3), 0, 1)
    strokes = 1 - dodge  # 0 = paper, 1 = heavy graphite
    rng = np.random.default_rng(0)
    noise = cv2.GaussianBlur(rng.random(g.shape).astype(np.float32), (0, 0), 0.8 * s + 0.3)
    noise = (noise - noise.min()) / max(np.ptp(noise), 1e-6)
    marks = strokes > (0.08 + p["grain"] * noise)
    return thicken(marks, p["weight"])


# ---------- entry point ----------

def render(name: str, rgb: np.ndarray, mask: np.ndarray, ink_light: bool,
           overrides: dict | None = None) -> np.ndarray:
    style = STYLES[name]
    p = {**style.defaults, **(overrides or {}), "ink_light": ink_light}
    if p.get("invert"):
        p["ink_light"] = not p["ink_light"]
    ink = style.fn(rgb, mask, p)
    return ink & mask if style.clip else ink
