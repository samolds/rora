"""Animated inputs (GIF, MP4, MOV): read frames, write ink frames back as an animation."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence

from .export import BACKGROUNDS, COLORS, to_rgba

EXTS = {".gif", ".mp4", ".mov"}


def is_animation(path: Path) -> bool:
    return path.suffix.lower() in EXTS


def _gif_frames(path: Path):
    with Image.open(path) as im:
        for frame in ImageSequence.Iterator(im):
            # browsers show 0 ms GIF frames for 100 ms
            yield np.asarray(frame.convert("RGB")), frame.info.get("duration") or 100


def _video_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"can't read video {path}")
    ms = 1000 / (cap.get(cv2.CAP_PROP_FPS) or 30)
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), ms
    finally:
        cap.release()


def _fit(rgb: np.ndarray, size: int) -> np.ndarray:
    f = size / max(rgb.shape[:2])
    if f == 1:
        return rgb
    return cv2.resize(rgb, None, fx=f, fy=f,
                      interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)


def read_frames(path: Path, size: int, fps: float = 0,
                max_frames: int = 0) -> tuple[list[np.ndarray], list[int]]:
    """RGB frames scaled so the long side is `size` px, and how long each shows (ms). Keeps
    at most `fps` frames per second (0 = all) and the first `max_frames` (0 = all)."""
    frames = _gif_frames(path) if path.suffix.lower() == ".gif" else _video_frames(path)
    step = 1000 / fps if fps else 0
    kept, starts, t = [], [], 0.0
    for rgb, ms in frames:
        if t + 0.5 >= len(starts) * step:
            if max_frames and len(starts) == max_frames:
                break
            kept.append(_fit(rgb, size))
            starts.append(t)
        t += ms
    durations = np.diff([*starts, t]).round().astype(int).tolist()
    return kept, durations


def save_gif(inks: list[np.ndarray], path: Path, color: str, durations: list[int],
             settings: str | None = None, halo: bool = False, opaque: bool = False,
             antialias: bool = False):
    """Transparent GIF with hard edges (GIF transparency is on/off); `halo` outlines the lines
    in the background colour so they show on any background, `opaque` draws them on it."""
    if opaque:
        # 256-step ramp from background to ink, indexed by the (antialiased) ink alpha
        ramp = np.linspace(BACKGROUNDS[color], COLORS[color], 256).round().astype(np.uint8)
        palette = ramp.ravel().tolist()
        indices = [np.ascontiguousarray(np.asarray(to_rgba(ink, color, antialias))[..., 3])
                   for ink in inks]
    else:
        # index 0 transparent, 1 ink, 2 halo
        palette = [*BACKGROUNDS[color], *COLORS[color], *BACKGROUNDS[color]]
        indices = [ink.astype(np.uint8) for ink in inks]
        if halo:
            r = max(2, round(max(inks[0].shape) / 240))  # 3 px at the default 720 px
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            for idx in indices:
                idx[(cv2.dilate(idx, kernel) > 0) & (idx == 0)] = 2
    frames = []
    for idx in indices:
        im = Image.fromarray(idx, "P")
        im.putpalette(palette)
        frames.append(im)
    extra = {} if opaque else dict(transparency=0, disposal=2)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   optimize=False, comment=settings or "", **extra)


def save_video(inks: list[np.ndarray], path: Path, color: str, durations: list[int],
               antialias: bool):
    """MP4/MOV on the shirt-like background (video has no transparency)."""
    fps = 1000 * len(durations) / max(sum(durations), 1)
    h, w = inks[0].shape
    size = (w + w % 2, h + h % 2)  # H.264 needs even sizes
    for codec in ("avc1", "mp4v"):  # avc1 (H.264) plays everywhere, if this OpenCV has it
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            break
    else:
        raise RuntimeError(f"can't write video {path}")
    bg = np.array(BACKGROUNDS[color], np.float32)[::-1]
    fg = np.array(COLORS[color], np.float32)[::-1]
    canvas = np.empty((size[1], size[0], 3), np.uint8)
    canvas[:] = bg.astype(np.uint8)
    try:
        for ink in inks:
            alpha = np.asarray(to_rgba(ink, color, antialias))[..., 3:] / 255.0
            canvas[:h, :w] = (bg + (fg - bg) * alpha).round().astype(np.uint8)
            writer.write(canvas)
    finally:
        writer.release()
