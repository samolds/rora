"""Optional final pass: run a finished drawing through an image-editing model (FLUX.2 [klein] 4B
by default, see MODELS) to clean it up or restyle it, inside this process.

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

from . import env, seams
from .segment import describe

log = logging.getLogger(__name__)

MODES = ("off", "auto", "clean", "stylize")
BACKENDS = ("auto", "mflux", "diffusers")


@dataclass(frozen=True)
class EditModel:
    """An image-editing model --refine can use. Sizes are the bf16 weights on disk (GB)."""
    key: str
    name: str
    repo: str
    revision: str
    licence: str
    family: str  # "klein" (FLUX.2 [klein] edit) or "kontext" (FLUX.1 Kontext)
    mflux_config: str  # ModelConfig factory in mflux
    steps: int
    guidance: float
    transformer_gb: float
    text_encoder_gb: float
    working_gb: float  # mflux working memory at ~1 MP on top of the weights
    required: tuple  # files that must be there before a run
    patterns: tuple  # what download() fetches
    measured: bool = False  # memory numbers measured on a 16 GB M2 Pro, else estimated
    gated: bool = False  # needs a Hugging Face login and accepting the licence on its page

    @property
    def download_gb(self) -> float:
        return self.transformer_gb + self.text_encoder_gb + 0.2


_KLEIN_FILES = ("model_index.json", "text_encoder/model.safetensors.index.json",
                "tokenizer/tokenizer.json", "vae/diffusion_pytorch_model.safetensors")
_KLEIN_PATTERNS = ("model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*",
                   "transformer/*", "vae/*")
MODELS = {m.key: m for m in (
    # revision: the commit mflux 0.23 pins, so both backends load the same single download
    EditModel("klein-4b", "FLUX.2 [klein] 4B", "black-forest-labs/FLUX.2-klein-4B",
              "e7b7dc27f91deacad38e78976d1f2b499d76a294", "Apache-2.0", "klein", "flux2_klein_4b",
              steps=4, guidance=1.0, transformer_gb=7.8, text_encoder_gb=8.0, working_gb=3.9,
              required=_KLEIN_FILES + ("transformer/diffusion_pytorch_model.safetensors",),
              patterns=_KLEIN_PATTERNS, measured=True),
    # revisions below: the repo's main branch on 2026-10-20
    EditModel("klein-9b", "FLUX.2 [klein] 9B", "black-forest-labs/FLUX.2-klein-9B",
              "92196c8e11f7b6cf2b7493e037d8c5345c559216", "FLUX non-commercial", "klein",
              "flux2_klein_9b", steps=4, guidance=1.0, transformer_gb=18.2, text_encoder_gb=16.4,
              working_gb=5.2,
              required=_KLEIN_FILES + ("transformer/diffusion_pytorch_model.safetensors.index.json",),
              patterns=_KLEIN_PATTERNS, gated=True),
    EditModel("kontext-dev", "FLUX.1 Kontext [dev] 12B", "black-forest-labs/FLUX.1-Kontext-dev",
              "24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d", "FLUX.1 [dev] non-commercial",
              "kontext", "dev_kontext", steps=28, guidance=2.5, transformer_gb=23.8,
              text_encoder_gb=9.7, working_gb=3.9,
              required=("model_index.json",
                        "transformer/diffusion_pytorch_model.safetensors.index.json",
                        "text_encoder/model.safetensors",
                        "text_encoder_2/model.safetensors.index.json",
                        "tokenizer/vocab.json", "tokenizer_2/spiece.model",
                        "vae/diffusion_pytorch_model.safetensors"),
              patterns=("model_index.json", "scheduler/*", "text_encoder/*", "text_encoder_2/*",
                        "tokenizer/*", "tokenizer_2/*", "transformer/*", "vae/*"), gated=True),
)}
DEFAULT_MODEL = "klein-4b"
MODEL_SIZE = 1024  # long side sent to the model; they work around 1 MP
MLX_CACHE_BYTES = 1 << 30  # mflux: cap on MLX's pool of freed buffers
BASELINE_GB = 1.5  # this process before the editing model loads (libraries, OWLv2 on the CPU)
OS_RESERVE_GB = 3.0  # what macOS and the window server need to stay responsive
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

def setup_help(m: "EditModel") -> str:
    login = ("\n     It is gated: log in first (env/bin/hf auth login) and accept its licence on\n"
             f"     https://huggingface.co/{m.repo}") if m.gated else ""
    flag = "" if m.key == DEFAULT_MODEL else f"={m.key}"
    return f"""\
How to set up the refine pass (one time):
  1. Install the Python packages (mflux on Apple silicon, diffusers elsewhere):
       env/bin/pip install -r requirements.txt
  2. Download {m.name} (about {m.download_gb:.0f} GB, {m.licence} licence) into the shared
     Hugging Face cache, {env.SHARED_HF_HUB} (other projects using mflux or
     diffusers find it there too; set RORA_REFINE_CACHE to use another folder):
       ./setup.sh --refine{flag}{login}
  Memory: with mflux, FLUX.2 [klein] 4B at the default 4-bit peaks around 8 GB in the refine
  phase; 8-bit and bigger models need more (each run checks before loading)."""


class RefineError(RuntimeError):
    pass


def _banner(headline: str, detail: str = "") -> str:
    bar = "=" * 78
    return f"\n{bar}\n {headline}\n{bar}\n{detail or setup_help(MODELS[DEFAULT_MODEL])}\n{bar}"


@dataclass
class Options:
    backend: str = "mflux"  # resolved by check_ready()
    prompt: str = ""  # extra instruction appended to the built-in one
    seed: int = 1
    threshold: float = 0.5  # auto: skip at or above this subject score; restyle below half of it
    quantize: int = 4  # mflux only: 4 or 8 bit, 0 = full precision
    model: str = DEFAULT_MODEL  # key in MODELS


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
    seams: tuple[int, int] = (0, 0)  # (breaks bridged, seam lines removed) by the seam pass
    settings: dict = field(default_factory=dict)


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def model_dir(m: "EditModel") -> Path:
    name = "models--" + m.repo.replace("/", "--")
    return env.SHARED_HF_HUB / name / "snapshots" / m.revision


def download(key: str = DEFAULT_MODEL):
    """Fetch a model into the shared Hugging Face cache (used by setup.sh --refine)."""
    from huggingface_hub import snapshot_download

    m = MODELS[key]
    snapshot_download(m.repo, revision=m.revision, allow_patterns=list(m.patterns),
                      cache_dir=env.SHARED_HF_HUB)


def _missing_files(m: "EditModel") -> list[str]:
    """Required files that aren't there, including shards an index lists (catches a
    download that stopped half way)."""
    import json

    root, missing = model_dir(m), []
    for f in m.required:
        if not (root / f).exists():
            missing.append(f)
        elif f.endswith(".index.json"):
            shards = set(json.loads((root / f).read_text())["weight_map"].values())
            missing += [f"{Path(f).parent}/{s}" for s in sorted(shards)
                        if not (root / Path(f).parent / s).exists()]
    return missing


def check_ready(backend: str, key: str = DEFAULT_MODEL) -> str:
    """Resolve `auto`, check the package and model are installed; returns the backend name."""
    m = MODELS[key]
    if backend == "auto":
        backend = "mflux" if is_apple_silicon() else "diffusers"
    if backend == "mflux" and not is_apple_silicon():
        raise RefineError(_banner(
            "--refine-backend mflux needs a Mac with Apple silicon; use diffusers here.",
            setup_help(m)))
    if importlib.util.find_spec(backend) is None:
        raise RefineError(_banner(f"--refine needs the '{backend}' package; it isn't installed.",
                                  setup_help(m)))
    missing = _missing_files(m)
    if missing:
        raise RefineError(_banner(
            f"--refine needs the {m.name} model, which isn't (fully) downloaded "
            f"(missing {missing[0]}).", setup_help(m)))
    return backend


def _weight_factor(quantize: int) -> float:
    """Size of quantized weights relative to bf16 (the extra is the per-group scales)."""
    return quantize / 16 + 0.03 if quantize else 1.0


def estimate_memory_gb(key: str, backend: str, quantize: int) -> float:
    """Rough peak memory of the refine phase for the whole process, in GB. Calibrated on
    FLUX.2 [klein] 4B with mflux (4-bit: 6.2 GB MLX peak + ~1.5 GB process, measured);
    the other models are scaled from it."""
    m = MODELS[key]
    f = _weight_factor(quantize) if backend == "mflux" else 1.0
    transformer, text_encoder, vae = m.transformer_gb * f, m.text_encoder_gb * f, 0.2
    if backend == "mflux" and m.family == "klein":
        # the text encoder is only loaded while it encodes the prompt (see below)
        peak = max(transformer + vae + m.working_gb, transformer + vae + text_encoder + 1.5)
    else:
        peak = transformer + vae + text_encoder + m.working_gb
    return BASELINE_GB + peak


def check_memory(key: str, backend: str, quantize: int, force: bool = False) -> str | None:
    """Guard rail run before anything loads. Raises RefineError if the model would not fit in
    this machine's RAM (heavy swapping can freeze or crash it) unless `force`; returns a warning
    if it fits but more than is free right now."""
    import psutil

    m = MODELS[key]
    need = estimate_memory_gb(key, backend, quantize)
    vm = psutil.virtual_memory()
    total, free = vm.total / 1e9, vm.available / 1e9
    bits = f", {quantize}-bit" if backend == "mflux" and quantize else ""
    measured = m.measured and backend == "mflux" and quantize == 4
    what = (f"{m.name} via {backend}{bits} needs about {need:.0f} GB "
            f"({'measured' if measured else 'estimated'})")
    if need > total - OS_RESERVE_GB:
        fixes = ([f"a smaller model (--refine-model {DEFAULT_MODEL})"] if key != DEFAULT_MODEL
                 else []) + (["--refine-quantize 4"] if backend == "mflux" and quantize != 4
                             else [])
        msg = (f"{what}, but this machine has {vm.total / 2**30:.0f} GB of RAM. It would swap "
               "heavily and could freeze or crash the machine. "
               + (f"Use {' or '.join(fixes)}, or add" if fixes else "Add")
               + " --refine-force to run anyway.")
        if not force:
            raise RefineError(_banner("--refine: model too big for this machine", msg))
        return msg
    if need > free:
        return (f"{what}; {free:.0f} GB is free right now, so macOS will compress or swap "
                "memory, which slows the run. Closing other apps helps.")
    return None


_editor = None


def _free_text_encoder_after_use(model, quantize: int | None, path: Path):
    """mflux, FLUX.2 [klein]: hold the text encoder (~2 GB for 4B at 4-bit) only while it encodes
    the prompt, so it isn't resident while the drawing is encoded, denoised and decoded. It is
    reloaded from disk (a few seconds) for the next drawing. Uses mflux 0.23 internals."""
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
        weights = WeightLoader.load_single_local(component, path)
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
    """Load the model once; returns (edit(canvas, prompt, seed) -> RGB array, tiled), where
    `tiled` says whether the VAE works in tiles (and so may leave seams)."""
    m = MODELS[opts.model]
    t = time.perf_counter()
    if opts.backend == "mflux":
        import mlx.core as mx
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.common.vae.tiling_config import TilingConfig

        # MLX keeps freed buffers for reuse with no cap by default; loading and quantizing the
        # bf16 weights alone leaves ~8 GB there, which pushes a 16 GB Mac into swap
        mx.set_cache_limit(MLX_CACHE_BYTES)
        config = getattr(ModelConfig, m.mflux_config)()
        quantize = opts.quantize or None
        if m.family == "klein":
            from mflux.models.flux2.variants import Flux2KleinEdit

            model = Flux2KleinEdit(quantize=quantize, model_path=str(model_dir(m)),
                                   model_config=config)
            _free_text_encoder_after_use(model, quantize, model_dir(m))
        else:
            from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext

            model = Flux1Kontext(quantize=quantize, model_path=str(model_dir(m)),
                                 model_config=config)
        # encoding/decoding the whole image at once needs up to ~5/~9 GB extra at 1 MP. Tiled
        # decoding leaves the ink unchanged; tiled encoding redraws small details (ink IoU ~0.9
        # against untiled) at the same quality; seams.repair() fixes what the tiles leave.
        model.tiling_config = TilingConfig(
            vae_decode_tile_size=seams.TILE, vae_decode_overlap=seams.OVERLAP // seams.SCALE,
            vae_encode_tile_size=seams.TILE, vae_encode_tile_overlap=seams.OVERLAP)

        def edit(canvas: np.ndarray, prompt: str, seed: int) -> np.ndarray:
            h, w = canvas.shape[:2]
            with tempfile.TemporaryDirectory(prefix="rora_refine_") as d:
                path = Path(d) / "drawing.png"  # mflux reads reference images from files
                Image.fromarray(canvas).save(path)
                image = {"image_paths": [path]} if m.family == "klein" else {"image_path": path}
                out = model.generate_image(seed=seed, prompt=prompt, num_inference_steps=m.steps,
                                           height=h, width=w, guidance=m.guidance, **image)
            return np.asarray(out.image.convert("RGB"))
        where, tiled = (f"mflux, {opts.quantize}-bit" if opts.quantize
                        else "mflux, full precision"), True
    else:
        import torch

        if m.family == "klein":
            from diffusers import Flux2KleinPipeline as Pipeline
        else:
            from diffusers import FluxKontextPipeline as Pipeline
        pipe = Pipeline.from_pretrained(str(model_dir(m)), torch_dtype=torch.bfloat16)
        if device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        def edit(canvas: np.ndarray, prompt: str, seed: int) -> np.ndarray:
            h, w = canvas.shape[:2]
            img = pipe(image=Image.fromarray(canvas), prompt=prompt, height=h, width=w,
                       num_inference_steps=m.steps, guidance_scale=m.guidance,
                       generator=torch.Generator("cpu").manual_seed(seed)).images[0]
            return np.asarray(img.convert("RGB"))
        where, tiled = f"diffusers on {device}", False
    log.info("  loaded %s (%s) in %.1fs", m.name, where, time.perf_counter() - t)
    return edit, tiled


def _generate(canvas: np.ndarray, prompt: str, opts: Options, device: str):
    """Returns (RGB result, whether the VAE worked in tiles)."""
    global _editor
    try:
        if _editor is None:
            _editor = _load_editor(opts, device)
        edit, tiled = _editor
        return edit(canvas, prompt, opts.seed), tiled
    except Exception as e:  # out of memory, missing files, ...: report it, keep the original
        raise RefineError(f"{MODELS[opts.model].name} via {opts.backend} failed: "
                          f"{type(e).__name__}: {e}") from e


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
    raw, tiled = _generate(sent, prompt, opts, device)
    # final pass: repair what the VAE's tiles can leave (broken strokes, straight lines)
    fixed, bridged, removed = seams.repair(raw) if tiled else (raw, 0, 0)

    size = max(fr.y1 - fr.y0, fr.x1 - fr.x0)
    r = int(0.02 * size) + 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    keep = cv2.dilate((mask | ink).astype(np.uint8), kernel) > 0
    new = from_canvas(fixed, fr, mask.shape, keep)
    after = subject_score(finder, to_canvas(new, fr), kinds)

    res = Result(mode, before, after, new, sent=sent, raw=raw, seams=(bridged, removed),
                 settings={"refine": mode, "refine_prompt": opts.prompt, "refine_seed": opts.seed,
                           "refine_subject": desc,
                           "refine_model": MODELS[opts.model].repo,
                           "refine_backend": opts.backend,
                           "refine_quantize": opts.quantize if opts.backend == "mflux" else None,
                           "refine_seams_bridged": bridged, "refine_seam_lines_removed": removed,
                           "refine_score_before": round(before, 3),
                           "refine_score_after": round(after, 3)})
    if new.sum() < 0.001 * mask.sum():
        res.ink, res.rejected = None, "the result is (almost) empty"
    elif after < before - SCORE_MARGIN:
        res.ink, res.rejected = None, f"it looks less like {desc} ({before:.2f} -> {after:.2f})"
    res.seconds = time.perf_counter() - t
    return res
