"""Turn an ink mask into print-ready files: transparent PNG, traced SVG, preview sheet."""
import html
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin, PngImagePlugin

COLORS = {"white": (255, 255, 255), "black": (0, 0, 0)}


def despeckle(ink: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return ink
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[labels]


def crop_and_resize(ink: np.ndarray, mask: np.ndarray, pad: float, size: int) -> np.ndarray:
    """Crop to the dog plus padding, then scale so the long side is `size` px (0 = keep)."""
    ys, xs = np.nonzero(mask)
    p = int(pad * max(np.ptp(ys), np.ptp(xs)))
    y0, y1 = max(ys.min() - p, 0), min(ys.max() + p + 1, ink.shape[0])
    x0, x1 = max(xs.min() - p, 0), min(xs.max() + p + 1, ink.shape[1])
    ink = ink[y0:y1, x0:x1]
    if size:
        f = size / max(ink.shape)
        resized = cv2.resize(ink.astype(np.float32), None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
        ink = resized > 0.5
    return ink


def to_rgba(ink: np.ndarray, color: str, antialias: bool) -> Image.Image:
    alpha = ink.astype(np.float32)
    if antialias:
        alpha = cv2.GaussianBlur(alpha, (0, 0), 0.7)
    rgba = np.zeros((*ink.shape, 4), np.uint8)
    rgba[..., :3] = COLORS[color]
    rgba[..., 3] = (alpha * 255).round().astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def save_png(ink, path: Path, color: str, antialias: bool, dpi: int,
             settings: str | None = None) -> Image.Image:
    img = to_rgba(ink, color, antialias)
    info = PngImagePlugin.PngInfo()
    if settings:
        info.add_text("dog_lineart_settings", settings)
    img.save(path, dpi=(dpi, dpi), pnginfo=info)
    return img


def save_svg(ink: np.ndarray, path: Path, color: str, simplify: float = 0.8,
             settings: str | None = None) -> int:
    """Trace the ink mask into filled paths (holes kept via even-odd fill). Returns path count."""
    contours, _ = cv2.findContours(ink.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    parts = []
    for c in contours:
        c = cv2.approxPolyDP(c, simplify, True)
        if len(c) < 3:
            continue
        pts = c[:, 0, :]
        parts.append("M" + "L".join(f"{x},{y}" for x, y in pts) + "Z")
    h, w = ink.shape
    hex_color = "#%02x%02x%02x" % COLORS[color]
    desc = f"<desc>{html.escape(settings)}</desc>\n" if settings else ""
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n'
        f'{desc}<path fill="{hex_color}" fill-rule="evenodd" d="{"".join(parts)}"/>\n</svg>\n'
    )
    return len(parts)


def save_preview(items: list[tuple[str, Image.Image]], path: Path, color: str, thumb: int = 600):
    """Contact sheet on a shirt-like background so you can judge the print."""
    if not items:
        return
    bg = (18, 18, 18) if color == "white" else (235, 235, 235)
    cols = min(4, len(items))
    rows = -(-len(items) // cols)
    label_h = 28
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), bg)
    draw = ImageDraw.Draw(sheet)
    for i, (name, img) in enumerate(items):
        t = img.copy()
        t.thumbnail((thumb - 20, thumb - 20))
        x = (i % cols) * thumb + (thumb - t.width) // 2
        y = (i // cols) * (thumb + label_h) + (thumb - t.height) // 2
        sheet.paste(t, (x, y), t)
        draw.text(((i % cols) * thumb + 10, (i // cols) * (thumb + label_h) + thumb + 4), name,
                  fill=(160, 160, 160))
    sheet.save(path)
