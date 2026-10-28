"""WhatsApp stickers: the line art as a 512x512 transparent WebP (animated for clips).

WhatsApp only treats a WebP as a sticker if it is exactly 512x512, at most 100 KB (500 KB
animated) and at most 10 s long, so clips are trimmed and the encoding is tightened to fit.
"""
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .export import BACKGROUNDS, COLORS

SIZE = 512
MARGIN = 16  # WhatsApp's recommended clear margin
HALO = 4  # px of background colour around the lines, so white ink shows on light chats
MAX_BYTES, MAX_ANIM_BYTES, MAX_MS = 100_000, 500_000, 10_000
MIN_FRAME_MS = 8
# tried in order until the file fits; the ink is the alpha channel, so it's alpha_quality
# (fewer alpha levels) that shrinks line art, and even 0 keeps the lines clean
ENCODINGS = [dict(lossless=True, quality=50, method=4),
             *(dict(quality=q, alpha_quality=aq, method=4) for q, aq in ((80, 50), (60, 20), (50, 0)))]


def frames(inks: list[np.ndarray], color: str) -> list[Image.Image]:
    """512x512 RGBA frames: all cropped to where any frame has ink (so a clip doesn't jump),
    scaled to fit inside the margin, with a thin halo in the background colour."""
    union = np.logical_or.reduce(inks)
    ys, xs = np.nonzero(union) if union.any() else np.nonzero(np.ones_like(union))
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    edge = MARGIN + HALO + 2  # + the halo's soft edge
    f = (SIZE - 2 * edge) / max(y1 - y0, x1 - x0)
    w, h = max(1, round((x1 - x0) * f)), max(1, round((y1 - y0) * f))
    ox, oy = (SIZE - w) // 2, (SIZE - h) // 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * HALO + 1, 2 * HALO + 1))
    fg = np.array(COLORS[color], np.float32)
    bg = np.array(BACKGROUNDS[color], np.float32)
    out = []
    for ink in inks:
        crop = ink[y0:y1, x0:x1].astype(np.float32)
        if f < 1:
            a = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
            a = np.clip(2 * a, 0, 1)  # lines thinner than a pixel after shrinking stay visible
        else:
            a = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        ink_a = np.zeros((SIZE, SIZE), np.float32)
        ink_a[oy:oy + h, ox:ox + w] = a
        halo_a = cv2.GaussianBlur(cv2.dilate(ink_a, kernel), (0, 0), 0.7)
        halo_a = np.maximum(halo_a, ink_a)
        alpha = ink_a + halo_a * (1 - ink_a)  # ink over halo
        rgb = (fg * ink_a[..., None] + bg * (halo_a * (1 - ink_a))[..., None]) \
            / np.maximum(alpha, 1e-6)[..., None]
        rgba = np.dstack([rgb, 255 * alpha]).round().clip(0, 255).astype(np.uint8)
        out.append(Image.fromarray(rgba, "RGBA"))
    return out


def _encode(images: list[Image.Image], durations: list[int] | None, opts: dict) -> bytes:
    buf = io.BytesIO()
    if durations:
        images[0].save(buf, "WEBP", save_all=True, append_images=images[1:],
                       duration=durations, loop=0, background=(0, 0, 0, 0), **opts)
    else:
        images[0].save(buf, "WEBP", **opts)
    return buf.getvalue()


def save(inks: list[np.ndarray], path: Path, color: str,
         durations: list[int] | None = None) -> str:
    """Write the sticker; `durations` (ms per frame) makes it animated. Returns a note on what
    was changed to fit WhatsApp's limits ('' if nothing), including if it still doesn't fit."""
    notes = []
    animated = durations is not None and len(inks) > 1
    if animated:
        starts = np.cumsum(durations) - np.asarray(durations)
        n = max(1, int((starts + MIN_FRAME_MS <= MAX_MS).sum()))
        if n < len(inks) or sum(durations) > MAX_MS:
            notes.append(f"trimmed to {MAX_MS / 1000:g} s")
        inks = inks[:n]
        durations = [max(MIN_FRAME_MS, int(d)) for d in durations[:n]]
        durations[-1] = min(durations[-1], MAX_MS - int(starts[n - 1]))
    else:
        inks, durations = inks[:1], None
    images = frames(inks, color)
    limit = MAX_ANIM_BYTES if animated else MAX_BYTES

    smallest, total = None, len(images)
    while True:
        for opts in ENCODINGS:
            data = _encode(images, durations, opts)
            if smallest is None or len(data) < len(smallest):
                smallest = data
            if len(data) <= limit:
                if len(images) < total:
                    notes.append(f"{len(images)} of {total} frames")
                if not opts.get("lossless"):
                    notes.append(f"lossy, alpha quality {opts['alpha_quality']}")
                path.write_bytes(data)
                return ", ".join(notes)
        if not durations or len(images) < 2:
            break
        # still too big: drop every other frame, keeping the input's frame rate (so it's shorter)
        images, durations = images[::2], durations[::2]
    path.write_bytes(smallest)
    notes.append(f"{len(smallest) // 1000} KB is over WhatsApp's {limit // 1000} KB limit")
    return ", ".join(notes)
