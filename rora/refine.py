"""Optional final pass: run a finished drawing through the FLUX.2 [klein] 4B image-editing model
to clean it up or restyle it, inside this process.

Backends: mflux (MLX, Apple silicon) or Hugging Face diffusers (PyTorch; CUDA, or MPS/CPU).
The drawing is sent as black ink on white, and the result is turned back into an ink mask
clipped to the subject, so colour, transparency, cropping and SVG export work as for the original.
"""
import importlib.util
import logging
import platform
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import env
from .segment import describe

log = logging.getLogger(__name__)

MODES = ("off", "auto", "clean", "stylize")
BACKENDS = ("auto", "mflux", "diffusers")
MODEL_NAME = "FLUX.2 [klein] 4B"
MODEL_REPO = "black-forest-labs/FLUX.2-klein-4B"
# the commit mflux 0.23 pins, so both backends load the same single download
MODEL_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
DOWNLOAD_PATTERNS = ["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*",
                     "transformer/*", "vae/*"]
REQUIRED_FILES = ["model_index.json", "transformer/diffusion_pytorch_model.safetensors",
                  "text_encoder/model.safetensors.index.json", "tokenizer/tokenizer.json",
                  "vae/diffusion_pytorch_model.safetensors"]
DOWNLOAD_GB = 16
STEPS = 4  # klein 4B is step-distilled
MODEL_SIZE = 1024  # long side sent to the model; it works around 1 MP
MLX_CACHE_BYTES = 1 << 30  # mflux: cap on MLX's pool of freed buffers
SCORE_MARGIN = 0.1  # reject a result whose subject score drops by more than this
# interior lines the stylize prompt asks for, per subject
FORM_LINES = {"dog": "ears, legs and fur direction", "cat": "ears, legs and fur direction",
              "person": "hair, hands and clothing folds"}

PROMPTS = {
    "clean": (
        "Clean up this black ink drawing of {desc}. Keep exactly the same subjects, poses, "
        "proportions, framing, drawing style and line thickness. Remove stray marks and tiny "
        "broken fragments, join lines that are broken and smooth wobbly lines. Black ink on a "
        "plain white background only: no shading, no fill, no colour, no text, no new details."
    ),
    "stylize": (
        "Redraw this black ink drawing of {desc} more clearly, in the same drawing style, poses, "
        "proportions and framing. Make it unmistakably {desc}: confident continuous outlines, "
        "clearly drawn eyes, noses and mouths, and only a few interior lines for {form}. Black "
        "ink on a plain white background only: no shading, no fill, no colour, no text."
    ),
}

SETUP_HELP = f"""\
How to set up the refine pass (one time):
  1. Install the Python packages (mflux on Apple silicon, diffusers elsewhere):
       env/bin/pip install -r requirements.txt
  2. Download {MODEL_NAME} (about {DOWNLOAD_GB} GB, Apache-2.0 licence) into the shared
     Hugging Face cache, {env.SHARED_HF_HUB} (other projects using mflux or
     diffusers find it there too; set RORA_REFINE_CACHE to use another folder):
       ./setup.sh --refine
  Memory: mflux at the default 4-bit peaks around 6 GB during the refine phase (8-bit around
  7.5 GB); diffusers needs about 16 GB (CUDA VRAM, or unified memory on a Mac)."""


class RefineError(RuntimeError):
    pass


def _banner(headline: str, detail: str = SETUP_HELP) -> str:
    bar = "=" * 78
    return f"\n{bar}\n {headline}\n{bar}\n{detail}\n{bar}"


@dataclass
class Options:
    backend: str = "mflux"  # resolved by check_ready()
    prompt: str = ""  # extra instruction appended to the built-in one
    seed: int = 1
    threshold: float = 0.5  # auto: skip at or above this subject score; restyle below half of it
    quantize: int = 4  # mflux only: 4 or 8 bit, 0 = full precision


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


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def model_dir() -> Path:
    name = "models--" + MODEL_REPO.replace("/", "--")
    return env.SHARED_HF_HUB / name / "snapshots" / MODEL_REVISION


def download():
    """Fetch the model into the shared Hugging Face cache (used by setup.sh --refine)."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_REPO, revision=MODEL_REVISION, allow_patterns=DOWNLOAD_PATTERNS,
                      cache_dir=env.SHARED_HF_HUB)


def check_ready(backend: str) -> str:
    """Resolve `auto`, check the package and model are installed; returns the backend name."""
    if backend == "auto":
        backend = "mflux" if is_apple_silicon() else "diffusers"
    if backend == "mflux" and not is_apple_silicon():
        raise RefineError(_banner(
            "--refine-backend mflux needs a Mac with Apple silicon; use diffusers here."))
    if importlib.util.find_spec(backend) is None:
        raise RefineError(_banner(f"--refine needs the '{backend}' package; it isn't installed."))
    missing = [f for f in REQUIRED_FILES if not (model_dir() / f).exists()]
    if missing:
        raise RefineError(_banner(
            f"--refine needs the {MODEL_NAME} model, which isn't downloaded yet "
            f"(missing {missing[0]})."))
    return backend


_editor = None


def _free_text_encoder_after_use(model, quantize: int | None):
    """mflux: hold the text encoder (~2 GB at 4-bit) only while it encodes the prompt, so it
    isn't resident while the drawing is encoded, denoised and decoded. It is reloaded from
    disk (a few seconds) for the next drawing. Uses mflux 0.23 internals."""
    import gc

    import mlx.core as mx
    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
    from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

    encode = model._encode_prompt_pair

    def load():
        component = next(c for c in Flux2KleinWeightDefinition.get_components()
                         if c.name == "text_encoder")
        weights = WeightLoader.load_single_local(component, model_dir())
        model.text_encoder = Qwen3TextEncoder(**model.model_config.text_encoder_overrides)
        WeightApplier.apply_and_quantize(weights, {"text_encoder": model.text_encoder}, quantize,
                                         Flux2KleinWeightDefinition)

    def encode_then_free(**kwargs):
        if model.text_encoder is None:
            load()
        out = encode(**kwargs)
        mx.eval([x for x in out if isinstance(x, mx.array)])
        model.text_encoder = None
        gc.collect()
        mx.clear_cache()
        return out

    model._encode_prompt_pair = encode_then_free


def _load_editor(opts: Options, device: str):
    """Load the model once; returns edit(canvas, prompt, seed) -> RGB array."""
    t = time.perf_counter()
    if opts.backend == "mflux":
        import mlx.core as mx
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.common.vae.tiling_config import TilingConfig
        from mflux.models.flux2.variants import Flux2KleinEdit

        # MLX keeps freed buffers for reuse with no cap by default; loading and quantizing the
        # bf16 weights alone leaves ~8 GB there, which pushes a 16 GB Mac into swap
        mx.set_cache_limit(MLX_CACHE_BYTES)
        model = Flux2KleinEdit(quantize=opts.quantize or None, model_path=str(model_dir()),
                               model_config=ModelConfig.flux2_klein_4b())
        # encoding/decoding the whole image at once needs up to ~5/~9 GB extra at 1 MP. Tiled
        # decoding leaves the ink unchanged; tiled encoding redraws small details (ink IoU ~0.9
        # against untiled) at the same quality, with no visible seams on line art.
        model.tiling_config = TilingConfig()
        _free_text_encoder_after_use(model, opts.quantize or None)

        def edit(canvas: np.ndarray, prompt: str, seed: int) -> np.ndarray:
            h, w = canvas.shape[:2]
            with tempfile.TemporaryDirectory(prefix="rora_refine_") as d:
                path = Path(d) / "drawing.png"  # mflux reads reference images from files
                Image.fromarray(canvas).save(path)
                out = model.generate_image(seed=seed, prompt=prompt, num_inference_steps=STEPS,
                                           height=h, width=w, guidance=1.0, image_paths=[path])
            return np.asarray(out.image.convert("RGB"))
        where = f"mflux, {opts.quantize}-bit" if opts.quantize else "mflux, full precision"
    else:
        import torch
        from diffusers import Flux2KleinPipeline

        pipe = Flux2KleinPipeline.from_pretrained(str(model_dir()), torch_dtype=torch.bfloat16)
        if device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        def edit(canvas: np.ndarray, prompt: str, seed: int) -> np.ndarray:
            h, w = canvas.shape[:2]
            img = pipe(image=Image.fromarray(canvas), prompt=prompt, height=h, width=w,
                       num_inference_steps=STEPS, guidance_scale=1.0,
                       generator=torch.Generator("cpu").manual_seed(seed)).images[0]
            return np.asarray(img.convert("RGB"))
        where = f"diffusers on {device}"
    log.info("  loaded %s (%s) in %.1fs", MODEL_NAME, where, time.perf_counter() - t)
    return edit


def _generate(canvas: np.ndarray, prompt: str, opts: Options, device: str) -> np.ndarray:
    global _editor
    try:
        if _editor is None:
            _editor = _load_editor(opts, device)
        return _editor(canvas, prompt, opts.seed)
    except Exception as e:  # out of memory, missing files, ...: report it, keep the original
        raise RefineError(f"{MODEL_NAME} via {opts.backend} failed: {type(e).__name__}: {e}") from e


@dataclass
class _Frame:
    """Where the subject's crop sits on the model canvas."""
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
    """Black ink on white, cropped to the subject and scaled onto a model-sized RGB canvas."""
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


def subject_score(finder, canvas: np.ndarray, kinds: list[str]) -> float:
    """How much OWLv2 thinks the drawing shows each kind in `kinds` (the weakest one counts;
    0 if below its detection floor)."""
    # OWLv2 barely reacts to crisp thin lines; a slight blur makes scores stable across line weights
    blurred = cv2.GaussianBlur(canvas, (0, 0), 2)
    scores = []
    for kind in dict.fromkeys(kinds):
        queries = {kind: [f"a drawing of a {kind}", f"a line drawing of a {kind}"]}
        dets = finder.detect(blurred, queries)
        scores.append(max((d.score for d in dets), default=0.0))
    return min(scores)


def refine(ink: np.ndarray, mask: np.ndarray, kinds: list[str], mode: str, finder, opts: Options,
           device: str) -> Result:
    """Run one drawing of `kinds` (e.g. ["person", "dog"]) through the editing model.
    `mode` is auto, clean or stylize."""
    t = time.perf_counter()
    desc = describe(kinds)
    fr = _frame(mask, MODEL_SIZE)
    sent = to_canvas(ink, fr)
    before = subject_score(finder, sent, kinds)
    if mode == "auto":
        if before >= opts.threshold:
            return Result("skip", before, seconds=time.perf_counter() - t)
        mode = "clean" if before >= opts.threshold / 2 else "stylize"

    form = "; ".join(dict.fromkeys(FORM_LINES[k] for k in kinds))
    prompt = PROMPTS[mode].format(desc=desc, form=form)
    if opts.prompt.strip():
        prompt += " " + opts.prompt.strip()
    raw = _generate(sent, prompt, opts, device)

    size = max(fr.y1 - fr.y0, fr.x1 - fr.x0)
    r = int(0.02 * size) + 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    keep = cv2.dilate((mask | ink).astype(np.uint8), kernel) > 0
    new = from_canvas(raw, fr, mask.shape, keep)
    after = subject_score(finder, to_canvas(new, fr), kinds)

    res = Result(mode, before, after, new, sent=sent, raw=raw,
                 settings={"refine": mode, "refine_prompt": opts.prompt, "refine_seed": opts.seed,
                           "refine_subject": desc,
                           "refine_model": MODEL_REPO, "refine_backend": opts.backend,
                           "refine_quantize": opts.quantize if opts.backend == "mflux" else None,
                           "refine_score_before": round(before, 3),
                           "refine_score_after": round(after, 3)})
    if new.sum() < 0.001 * mask.sum():
        res.ink, res.rejected = None, "the result is (almost) empty"
    elif after < before - SCORE_MARGIN:
        res.ink, res.rejected = None, f"it looks less like {desc} ({before:.2f} -> {after:.2f})"
    res.seconds = time.perf_counter() - t
    return res
