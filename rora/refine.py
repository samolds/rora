"""Optional final pass: send a finished drawing through an image-editing model running in the
Draw Things app (https://drawthings.ai) to clean it up or restyle it.

Uses the app's local HTTP API (Automatic1111-compatible `/sdapi/v1/img2img`). The drawing is
sent as black ink on white, and the result is turned back into an ink mask clipped to the dog,
so colour, transparency, cropping and SVG export work exactly as for the original.
"""
import base64
import io
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:7860"
MODES = ("off", "auto", "clean", "stylize")
MODEL_SIZE = 1024  # long side sent to the model; the FLUX/Qwen edit models work around 1 MP
TIMEOUT = 900  # seconds per generation; big models on smaller Macs can take minutes
SCORE_MARGIN = 0.1  # reject a result whose dog score drops by more than this
# substrings of Draw Things model files that take the input image as an editing reference
EDIT_MODEL_HINTS = ("kontext", "edit", "klein", "flux_2", "flux_3")
DOG_QUERIES = {"dog": ["a drawing of a dog", "a line drawing of a dog"]}

PROMPTS = {
    "clean": (
        "Clean up this black ink drawing of a dog. Keep exactly the same dog, pose, proportions, "
        "framing, drawing style and line thickness. Remove stray marks and tiny broken fragments, "
        "join lines that are broken and smooth wobbly lines. Black ink on a plain white "
        "background only: no shading, no fill, no colour, no text, no new details."
    ),
    "stylize": (
        "Redraw this black ink drawing of a dog more clearly, in the same drawing style, pose, "
        "proportions and framing. Make it unmistakably a dog: a confident continuous outline, "
        "clearly drawn eyes, nose and mouth, and only a few interior lines for ears, legs and fur "
        "direction. Black ink on a plain white background only: no shading, no fill, no colour, "
        "no text."
    ),
}

SETUP_HELP = f"""\
How to set up Draw Things (free; macOS on Apple silicon):
  1. Install the app from the Mac App Store:
       https://apps.apple.com/us/app/draw-things-ai-generation/id6444050820
     (more at https://drawthings.ai)
  2. Open it, click the model name at the top of the settings panel, and download an
     image-editing model, for example:
       - FLUX.1 Kontext [dev]       good default
       - Qwen Image Edit 2511       best at keeping the drawing as is; bigger and slower
       - FLUX.2 [klein] 4B          smallest and fastest; a little less faithful
  3. Select that model, then turn on the API server in the app's settings
     ("API Server": protocol HTTP, port {urllib.parse.urlsplit(DEFAULT_URL).port}).
     Keep Draw Things open while the pipeline runs.
  4. Run again with --refine (add --refine-url if you chose another address or port)."""


class RefineError(RuntimeError):
    pass


def _banner(headline: str, detail: str = SETUP_HELP) -> str:
    bar = "=" * 78
    return f"\n{bar}\n {headline}\n{bar}\n{detail}\n{bar}"


@dataclass
class Options:
    url: str = DEFAULT_URL
    prompt: str = ""  # extra instruction appended to the built-in one
    seed: int = 1
    strength: float = 1.0  # 1.0 for editing models; 0.3-0.6 for plain image-to-image models
    threshold: float = 0.5  # auto: skip at or above this dog score; restyle below half of it


@dataclass
class Result:
    mode: str  # "skip", "clean" or "stylize"
    score_before: float
    score_after: float | None = None
    ink: np.ndarray | None = None  # None when skipped or rejected
    rejected: str | None = None
    sent: np.ndarray | None = None  # RGB image sent to the model (for --debug)
    raw: np.ndarray | None = None  # RGB image the model returned (for --debug)
    seconds: float = 0.0
    settings: dict = field(default_factory=dict)


def _request(url: str, payload: dict | None, timeout: float) -> dict:
    if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
        raise RefineError(f"--refine-url must start with http:// or https:// (got {url!r})")
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (scheme checked)
        return json.loads(resp.read())


def check_server(url: str, strength: float) -> str:
    """Make sure Draw Things answers at `url` with a usable model; returns the model file."""
    url = url.rstrip("/")
    try:
        cfg = _request(url + "/", None, timeout=5)
    except urllib.error.HTTPError as e:
        raise RefineError(_banner(
            f"Something answered at {url}, but not like Draw Things (HTTP {e.code}).")) from e
    except (urllib.error.URLError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise RefineError(_banner(
            f"--refine needs Draw Things, but nothing answered at {url} ({reason}).")) from e
    except ValueError as e:
        raise RefineError(_banner(
            f"Something answered at {url}, but not like Draw Things (no JSON settings).")) from e

    missing = {"prompt", "seed", "width", "height", "strength"} - set(cfg)
    if missing:
        raise RefineError(_banner(
            f"The server at {url} doesn't look like Draw Things (missing {sorted(missing)}).",
            "Point --refine-url at the Draw Things HTTP API server.\n\n" + SETUP_HELP))
    model = str(cfg.get("model") or "?")
    if strength >= 1 and not any(h in model.lower() for h in EDIT_MODEL_HINTS):
        raise RefineError(_banner(
            f"Draw Things is running, but the selected model ({model}) is not an image-editing "
            "model.",
            "At --refine-strength 1.0 a plain text-to-image model ignores the drawing entirely.\n"
            "Select an editing model in Draw Things (see step 2 below), or pass\n"
            "--refine-strength 0.3 .. 0.6 to use this model as plain image-to-image.\n\n"
            + SETUP_HELP))
    return model


def _encode_png(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _decode_image(data: str) -> np.ndarray:
    if data.startswith("data:"):
        data = data.partition(",")[2]
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))


def _generate(url: str, canvas: np.ndarray, prompt: str, opts: Options) -> np.ndarray:
    h, w = canvas.shape[:2]
    payload = {"prompt": prompt, "seed": opts.seed, "width": w, "height": h,
               "strength": opts.strength, "batch_count": 1, "init_images": [_encode_png(canvas)]}
    try:
        resp = _request(url.rstrip("/") + "/sdapi/v1/img2img", payload, TIMEOUT)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise RefineError(f"Draw Things rejected the request (HTTP {e.code}): {detail}") from e
    except (urllib.error.URLError, OSError) as e:
        raise RefineError(_banner(
            f"Lost the connection to Draw Things at {url} ({getattr(e, 'reason', e)}).")) from e
    images = resp.get("images") or []
    if not images:
        raise RefineError(f"Draw Things returned no image: {str(resp)[:300]}")
    return _decode_image(images[0])


@dataclass
class _Frame:
    """Where the dog's crop sits on the model canvas."""
    y0: int
    y1: int
    x0: int
    x1: int
    ox: int
    oy: int
    w: int
    h: int


def _frame(mask: np.ndarray, size: int) -> _Frame:
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    p = int(0.06 * max(y1 - y0, x1 - x0))
    y0, x0 = max(0, y0 - p), max(0, x0 - p)
    y1, x1 = min(mask.shape[0], y1 + p), min(mask.shape[1], x1 + p)
    f = size / max(y1 - y0, x1 - x0)
    w, h = max(1, round((x1 - x0) * f)), max(1, round((y1 - y0) * f))
    cw, ch = -(-w // 64) * 64, -(-h // 64) * 64  # canvas sides must be multiples of 64
    return _Frame(y0, y1, x0, x1, (cw - w) // 2, (ch - h) // 2, w, h)


def to_canvas(ink: np.ndarray, fr: _Frame) -> np.ndarray:
    """Black ink on white, cropped to the dog and scaled onto a model-sized RGB canvas."""
    crop = ink[fr.y0:fr.y1, fr.x0:fr.x1].astype(np.float32)
    small = cv2.resize(crop, (fr.w, fr.h), interpolation=cv2.INTER_AREA)
    ch, cw = -(-fr.h // 64) * 64, -(-fr.w // 64) * 64
    canvas = np.full((ch, cw), 255, np.uint8)
    canvas[fr.oy:fr.oy + fr.h, fr.ox:fr.ox + fr.w] = (255 * (1 - small)).round().astype(np.uint8)
    return np.repeat(canvas[..., None], 3, axis=2)


def from_canvas(rgb: np.ndarray, fr: _Frame, shape: tuple[int, int], keep: np.ndarray) -> np.ndarray:
    """Model output -> ink mask at working size, clipped to the `keep` region."""
    ch, cw = -(-fr.h // 64) * 64, -(-fr.w // 64) * 64
    if rgb.shape[:2] != (ch, cw):
        rgb = cv2.resize(rgb, (cw, ch), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[fr.oy:fr.oy + fr.h, fr.ox:fr.ox + fr.w]
    big = cv2.resize(gray, (fr.x1 - fr.x0, fr.y1 - fr.y0), interpolation=cv2.INTER_LINEAR)
    out = np.zeros(shape, bool)
    out[fr.y0:fr.y1, fr.x0:fr.x1] = big < 128
    return out & keep


def dog_score(finder, canvas: np.ndarray) -> float:
    """How much OWLv2 thinks the drawing shows a dog (0 if below its detection floor)."""
    # OWLv2 barely reacts to crisp thin lines; a slight blur makes scores stable across line weights
    dets = finder.detect(cv2.GaussianBlur(canvas, (0, 0), 2), DOG_QUERIES)
    return max((d.score for d in dets), default=0.0)


def refine(ink: np.ndarray, mask: np.ndarray, mode: str, finder, opts: Options) -> Result:
    """Run one drawing through Draw Things. `mode` is auto, clean or stylize."""
    t = time.perf_counter()
    fr = _frame(mask, MODEL_SIZE)
    sent = to_canvas(ink, fr)
    before = dog_score(finder, sent)
    if mode == "auto":
        if before >= opts.threshold:
            return Result("skip", before, seconds=time.perf_counter() - t)
        mode = "clean" if before >= opts.threshold / 2 else "stylize"

    prompt = PROMPTS[mode] + (" " + opts.prompt.strip() if opts.prompt.strip() else "")
    raw = _generate(opts.url, sent, prompt, opts)

    size = max(fr.y1 - fr.y0, fr.x1 - fr.x0)
    r = int(0.02 * size) + 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    keep = cv2.dilate((mask | ink).astype(np.uint8), kernel) > 0
    new = from_canvas(raw, fr, mask.shape, keep)
    after = dog_score(finder, to_canvas(new, fr))

    res = Result(mode, before, after, new, sent=sent, raw=raw,
                 settings={"refine": mode, "refine_prompt": opts.prompt, "refine_seed": opts.seed,
                           "refine_strength": opts.strength,
                           "refine_score_before": round(before, 3),
                           "refine_score_after": round(after, 3)})
    if new.sum() < 0.001 * mask.sum():
        res.ink, res.rejected = None, "the result is (almost) empty"
    elif after < before - SCORE_MARGIN:
        res.ink, res.rejected = None, f"it looks less like a dog ({before:.2f} -> {after:.2f})"
    res.seconds = time.perf_counter() - t
    return res
