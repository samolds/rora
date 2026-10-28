#!/usr/bin/env python3
"""Photo of a dog, cat or person -> transparent PNG line art (for printing on shirts).
GIFs and videos (MP4, MOV) are drawn frame by frame into a GIF (and, for videos, a video in the
input's format) of the line art."""
import argparse
import hashlib
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from rora import env  # noqa: F401  (must come before torch/ultralytics imports)
import cv2
import numpy as np

from rora import export, refine, styles, tattoo, track, video
from rora.segment import (COCO_KINDS, SUBJECTS, Segmenter, describe, free_device_memory,
                          load_image)

log = logging.getLogger("lineart")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tif", ".tiff"}

# Default --set values; any --set KEY=VALUE on the command line overrides these.
# They tune the tattoo looks (fineline, brush, ...) and are ignored by styles that don't use them.
DEFAULT_SETTINGS = {"max_strokes": 50, "head_strokes": 100, "form_lines": 100,
                    "min_votes": 2.5, "coverage": 0.1, "min_score": 0, "freedom": 0.4}
DEFAULT_DETAIL = 0.5

# command-line options that change the output image (others, like --svg, don't)
OUTPUT_OPTIONS = ("color", "size", "dpi", "pad", "min_speck", "outline", "antialias",
                  "work_size", "min_conf", "detector", "segmenter", "subject",
                  "anim_size", "fps", "max_frames", "steady")
STILL_ONLY = ("size", "dpi", "pad", "work_size")
ANIMATION_ONLY = ("anim_size", "fps", "max_frames", "steady")

# short names used in output file names (README has the same table)
ABBREV = {
    # all styles
    "detail": "d", "weight": "w", "invert": "inv",
    # line / tonal / cartoon styles
    "threshold": "th", "contrast": "ct", "epsilon": "eps", "blur": "bl", "min_length": "ml",
    "cell": "cl", "angle": "a", "spacing": "sp", "dot": "dt", "seed": "sd", "crosshatch": "xh",
    "levels": "lv", "light": "lt", "rim": "rm", "light_angle": "la", "grain": "gr",
    # tattoo looks
    "form_lines": "fl", "max_strokes": "bs", "head_strokes": "hs", "min_votes": "mv",
    "coverage": "cv", "tolerance": "tol", "min_score": "ms", "join_gap": "jg", "smooth": "sm",
    "edge_band": "eb", "freedom": "f", "features": "ft", "almond_eyes": "ae", "pupils": "pu",
    "nose_solid": "ns", "max_blobs": "mb", "darkness": "dk", "outline_ratio": "or",
    "taper": "tp", "passes": "ps", "wobble": "wb", "gaps": "gp", "gap_size": "gs",
    # output options
    "color": "col", "size": "px", "dpi": "dpi", "pad": "pad", "min_speck": "spk",
    "outline": "ol", "antialias": "aa", "work_size": "ws", "min_conf": "mc",
    "detector": "det", "segmenter": "seg", "anim_size": "as", "fps": "fps", "max_frames": "mf",
    "steady": "sty",
    # refine pass (only in refined files' names)
    "refine": "rf", "refine_seed": "rsd", "refine_quantize": "rq", "refine_prompt": "rp",
    "refine_model": "rmd",
}
assert len(set(ABBREV.values())) == len(ABBREV), "file-name abbreviations must be unique"


def fmt(key: str, value) -> str:
    key = ABBREV.get(key, key)
    if isinstance(value, bool):
        return key if value else f"{key}0"
    if isinstance(value, (int, float)):
        return f"{key}{float(value):g}"
    return f"{key}-{value}".replace("/", "-").replace(" ", "-")


def settings_id(name: str, overrides: dict, args, animated: bool = False) -> tuple[str, str]:
    """(file name settings part, full settings JSON) for one style.

    The settings part is the style, its detail and weight, then any other setting that
    differs from the normal run's defaults (the style's own, with DEFAULT_SETTINGS applied),
    all in short form (ABBREV), e.g. `edges_d0.4_w1` or `fineline_d0.5_w3_f0.7`. Only
    settings the style uses count, so the same effective settings always give the same name.
    `animated`: for GIFs and videos, which use the animation options instead of the print ones.
    """
    style = styles.STYLES[name]
    used = {k: v for k, v in {**style.defaults, **overrides}.items()
            if k in style.defaults or k == "invert"}
    unused = STILL_ONLY if animated else ANIMATION_ONLY
    out_opts = {k: getattr(args, k) for k in OUTPUT_OPTIONS if k not in unused}
    cli_defaults = vars(parse_args([]))
    baseline = {**style.defaults, **{k: v for k, v in DEFAULT_SETTINGS.items() if k in style.defaults}}

    parts = [name, fmt("detail", used["detail"]), fmt("weight", used["weight"])]
    parts += [fmt(k, v) for k, v in sorted(used.items())
              if k not in ("detail", "weight") and v != baseline.get(k)]
    # the subject shows up in file names as a suffix instead (see subject_groups)
    parts += [fmt(k, v) for k, v in sorted(out_opts.items())
              if v != cli_defaults[k] and k != "subject"]

    full = {"style": name, **used, **out_opts}
    full = {k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
            for k, v in full.items()}
    return "_".join(parts), json.dumps(full, sort_keys=True)


def refine_tag(mode: str, args) -> str:
    """File name suffix for a refined result, e.g. `rf-clean` or `rf-stylize_rsd7`."""
    cli_defaults = vars(parse_args([]))
    parts = [fmt("refine", mode)]
    parts += [fmt(k, getattr(args, k)) for k in ("refine_model", "refine_seed", "refine_quantize")
              if getattr(args, k) != cli_defaults[k]]
    if args.refine_prompt.strip():
        digest = hashlib.sha1(args.refine_prompt.strip().encode()).hexdigest()[:6]
        parts.append(fmt("refine_prompt", digest))
    return "_".join(parts)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="*", type=Path, default=[Path("img/in")],
                    help="image, GIF or video (MP4, MOV) files, or folders (default: img/in)")
    ap.add_argument("-o", "--out", type=Path, default=Path("img/out"))
    ap.add_argument("--sample", type=positive_int, metavar="N",
                    help="run only N images picked at random from the inputs (the seed is logged)")
    ap.add_argument("--sample-seed", type=int, metavar="SEED",
                    help="seed for --sample, to pick the same images again")
    ap.add_argument("--list-styles", action="store_true", help="show styles and their settings")

    g = ap.add_argument_group("style")
    g.add_argument("-s", "--style", nargs="+", default=["edges"],
                   help="one or more style names, 'tattoo' (all tattoo looks) or 'all' (default: edges)")
    g.add_argument("--detail", type=float, help="0 (simple) .. 1 (busy), default 0.5")
    g.add_argument("--weight", type=int, help="line weight in px")
    g.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any style setting, e.g. --set threshold=0.45 (defaults: "
                        + " ".join(f"{k}={v}" for k, v in DEFAULT_SETTINGS.items()) + ")")
    g.add_argument("--style-defaults", action="store_true",
                   help="ignore the command-line defaults above (--detail 0.5 and the --set "
                        "defaults) and use each style's own defaults (see --list-styles)")
    g.add_argument("--invert", action="store_true",
                   help="flip tonal styles (ink where the subject is dark)")

    g = ap.add_argument_group("output")
    g.add_argument("--color", choices=["white", "black"], default="white", help="ink colour")
    g.add_argument("--size", type=int, default=3600, help="long side in px (0 = no resize)")
    g.add_argument("--dpi", type=int, default=300)
    g.add_argument("--pad", type=float, default=0.03, help="padding around the subject, fraction")
    g.add_argument("--min-speck", type=int, default=12, help="drop specks smaller than this (px)")
    g.add_argument("--outline", type=int, default=0, help="add a silhouette outline this wide (px)")
    g.add_argument("--antialias", action="store_true", help="soft edges (hard by default)")
    g.add_argument("--svg", action="store_true", help="also write traced .svg files")
    g.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True,
                   help="write _preview.png contact sheet (default: on)")
    g.add_argument("--debug", action="store_true", help="save mask and cut-out per photo")

    g = ap.add_argument_group("detection")
    g.add_argument("--work-size", type=int, default=2500, help="processing size, long side px")
    g.add_argument("--min-conf", type=float, default=0.35, help="detection confidence")
    g.add_argument("--subject", choices=SUBJECTS, default="all",
                   help="what to draw: all = every person, cat and dog detected with at least "
                        "--subject-conf confidence, together in one drawing; auto = only the "
                        "main one (biggest, most central), adds e.g. _dog to the file name; "
                        "person / cat / dog = each one of that kind in its own file (_dog, or "
                        "_dog1, _dog2 ... if there are several) (default: all)")
    g.add_argument("--subject-conf", type=float, default=0.8,
                   help="confidence needed to draw subjects besides the main one, which is "
                        "always drawn (default: 0.8)")
    g.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    g.add_argument("--detector", default="yolo11x.pt")
    g.add_argument("--segmenter", default="sam2.1_b.pt")

    g = ap.add_argument_group("animation (GIF, MP4 and MOV inputs; drawn on the whole frame, "
                              "written in the input's format, plus a transparent GIF for videos; "
                              "no --refine, --svg or --debug)")
    g.add_argument("--anim-size", type=int, default=720,
                   help="long side in px, for processing and output (instead of --work-size "
                        "and --size) (default: 720)")
    g.add_argument("--fps", type=float, default=12,
                   help="draw at most this many frames per second, dropping the rest "
                        "(0 = every frame) (default: 12)")
    g.add_argument("--max-frames", type=int, default=0,
                   help="draw only the first N frames (after --fps) (default: 0 = all)")
    g.add_argument("--steady", type=float, default=0.5,
                   help="a subject detected in more than this fraction of the frames is drawn "
                        "in every frame, cut out again where detection missed it, so it doesn't "
                        "blink in and out (1 = off, every frame on its own) (default: 0.5)")

    g = ap.add_argument_group("refine (optional final pass through an image-editing model)")
    g.add_argument("--refine", choices=refine.MODES, default="off",
                   help="clean = light touch-up, stylize = redraw more clearly, auto = skip "
                        "drawings that already look like their subject, otherwise clean or stylize "
                        "(default: off). Writes the refined file next to the original")
    g.add_argument("--refine-model", choices=list(refine.MODELS), default=refine.DEFAULT_MODEL,
                   help="; ".join(f"{k} = {m.name} ({m.licence})"
                                  for k, m in refine.MODELS.items())
                        + f" (default: {refine.DEFAULT_MODEL}). Before loading, the run checks "
                          "the model fits in memory")
    g.add_argument("--refine-force", action="store_true",
                   help="run even if the model is too big for this machine's RAM (it may "
                        "swap heavily, freeze or crash)")
    g.add_argument("--refine-backend", choices=refine.BACKENDS, default="auto",
                   help="auto = mflux on Apple silicon, diffusers elsewhere")
    g.add_argument("--refine-quantize", type=int, choices=(0, 4, 8), default=4,
                   help="mflux weight precision in bits: 8 and 0 (full precision) need much more "
                        "memory, see the check at the start of the run (default: 4)")
    g.add_argument("--refine-prompt", default="",
                   help="extra instruction added to the built-in one, e.g. 'thicker outline'")
    g.add_argument("--refine-seed", type=int, default=1)
    g.add_argument("--refine-threshold", type=float, default=0.5,
                   help="auto: skip drawings whose subject score is at least this; below half of "
                        "it, stylize instead of clean (scores are in run.log)")
    return ap.parse_args(argv)


def parse_overrides(args) -> dict:
    out = {} if args.style_defaults else dict(DEFAULT_SETTINGS)
    for item in args.set:
        key, _, value = item.partition("=")
        try:
            out[key] = float(value) if "." in value else int(value)
        except ValueError:
            out[key] = value
    if args.detail is not None:
        out["detail"] = args.detail
    elif not args.style_defaults:
        out["detail"] = DEFAULT_DETAIL
    if args.weight is not None:
        out["weight"] = args.weight
    if args.invert:
        out["invert"] = True
    return out


def positive_int(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return n


def sample_images(files: list[Path], n: int, seed: int | None) -> list[Path]:
    """N files picked at random (in their original order); all of them if there are only N."""
    if seed is None:
        seed = random.randrange(2**31)
    if n >= len(files):
        log.info("--sample %d: only %d image(s), using all of them", n, len(files))
        return files
    picked = set(random.Random(seed).sample(range(len(files)), n))
    files = [f for i, f in enumerate(files) if i in picked]
    log.info("--sample %d (--sample-seed %d): %s", n, seed, ", ".join(f.name for f in files))
    return files


def collect_images(inputs: list[Path]) -> list[Path]:
    files = []
    for p in inputs:
        if p.is_dir():
            files += sorted(f for f in p.iterdir()
                            if f.suffix.lower() in IMAGE_EXTS | video.EXTS)
        elif p.is_file():
            files.append(p)
        else:
            log.warning("Not found: %s", p)
    return files


def setup_logging(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / "run.log")):
        h.setFormatter(fmt)
        root.addHandler(h)


def list_styles():
    for name, s in styles.STYLES.items():
        settings = ", ".join(f"{k}={v}" for k, v in s.defaults.items())
        print(f"{name:10s} {s.description}\n{'':10s} {settings}")


def save_outputs(args, ink: np.ndarray, stem: str, settings: str):
    """Write <stem>.png (and .svg); returns the PNG image and a log message."""
    img = export.save_png(ink, args.out / f"{stem}.png", args.color, args.antialias, args.dpi,
                          settings)
    msg = f"{stem}.png {img.width}x{img.height}"
    if args.svg:
        n = export.save_svg(ink, args.out / f"{stem}.svg", args.color, settings=settings)
        msg += f" + .svg ({n} paths)"
    return img, msg


def subject_groups(subjects: list, mode: str) -> list[tuple[str, list]]:
    """(file name suffix, subjects drawn together) per output: `all` draws everything in one
    file with no suffix; `auto` and a kind add the kind, numbered if there are several."""
    if mode == "all":
        return [("", subjects)]
    if len(subjects) == 1:
        return [(f"_{subjects[0].kind}", subjects)]
    return [(f"_{s.kind}{i}", [s]) for i, s in enumerate(subjects, 1)]


def draw_subjects(args, photo: str, rgb, subjects, names, ids, overrides, refine_jobs,
                  previews: list) -> int:
    """Draw the subjects together in every style; returns files written. With --refine, also
    queues each drawing in `refine_jobs` for the refine phase."""
    mask = np.logical_or.reduce([s.mask for s in subjects])
    kinds, written = [s.kind for s in subjects], 0
    if args.debug:
        cut = rgb.copy()
        cut[~mask] = 0
        cv2.imwrite(str(args.out / f"{photo}_debug_mask.png"), mask.astype(np.uint8) * 255)
        cv2.imwrite(str(args.out / f"{photo}_debug_cutout.jpg"),
                    cv2.cvtColor(cut, cv2.COLOR_RGB2BGR))

    # each subject is drawn on its own (own face, own outline) and the drawings are combined;
    # subjects on the outside so the tattoo passes' per-subject cache is reused across styles
    inks = {name: np.zeros(mask.shape, bool) for name in names}
    secs = dict.fromkeys(names, 0.0)
    for s in subjects:
        for name in names:
            t = time.perf_counter()
            inks[name] |= styles.render(name, rgb, s.mask, args.color == "white",
                                        {**overrides, "subject": s.kind})
            secs[name] += time.perf_counter() - t

    for name in names:
        t = time.perf_counter() - secs[name]
        ink = inks[name]
        if args.outline:
            ink |= styles.silhouette(mask, args.outline)
        ink = export.despeckle(ink, args.min_speck)
        out = export.crop_and_resize(ink, mask, args.pad, args.size)

        tag, settings = ids[name]
        settings = json.dumps({**json.loads(settings), "subjects": describe(kinds)},
                              sort_keys=True)
        stem = f"{photo}__{tag}"
        img, msg = save_outputs(args, out, stem, settings)
        log.info("  %s: %s, ink %.0f%% (%.1fs)", name, msg, 100 * out.mean(),
                 time.perf_counter() - t)
        previews.append((stem, img))
        written += 1
        if refine_jobs is not None:
            refine_jobs.append(RefineJob(stem, settings, kinds, mask.shape, np.packbits(ink),
                                         np.packbits(mask)))
    return written


def draw_animation(args, path: Path, seg, names, ids, overrides, previews: list) -> int:
    """Draw a GIF or video frame by frame on its whole frame, one animation per style in the
    input's format plus a transparent GIF for videos; returns files written (0 if no subject
    was found in any frame)."""
    frames, durations = video.read_frames(path, args.anim_size, args.fps, args.max_frames)
    if not frames:
        raise ValueError("no frames")
    h, w = frames[0].shape[:2]
    log.info("  %d frame(s) at %dx%d, %.1f fps", len(frames), w, h,
             1000 * len(frames) / max(sum(durations), 1))

    inks = {name: [] for name in names}
    rora_log = logging.getLogger("rora")
    level = rora_log.level
    rora_log.setLevel(logging.WARNING)  # the per-frame steps' log lines would repeat every frame
    try:
        found = []
        for i, rgb in enumerate(frames, 1):
            t = time.perf_counter()
            found.append(seg.find_subjects(rgb, args.subject, args.subject_conf))
            log.info("  frame %d/%d: %s (%.1fs)", i, len(frames),
                     describe([s.kind for s in found[-1]]) if found[-1] else "nothing found",
                     time.perf_counter() - t)
            free_device_memory()
        if not any(found):
            log.warning("  no %s found in any frame, skipping", args.subject
                        if args.subject in COCO_KINDS.values() else "person, cat or dog")
            return 0

        if args.steady < 1:
            t = time.perf_counter()
            found, filled = track.steady(found, lambda i, boxes: seg.segment(frames[i], boxes),
                                         args.steady)
            for label, at in filled.items():
                log.info("  kept the %s in %d more frame(s): %s", label, len(at),
                         ", ".join(str(i + 1) for i in at))
            if filled:
                log.info("  steadied in %.1fs", time.perf_counter() - t)
            free_device_memory()

        for i, (rgb, subjects) in enumerate(zip(frames, found), 1):
            t = time.perf_counter()
            mask = np.zeros((h, w), bool)
            frame = {name: np.zeros((h, w), bool) for name in names}
            for s in subjects:
                mask |= s.mask
                for name in names:
                    frame[name] |= styles.render(name, rgb, s.mask, args.color == "white",
                                                 {**overrides, "subject": s.kind})
            for name in names:
                ink = frame[name]
                if args.outline:
                    ink |= styles.silhouette(mask, args.outline)
                inks[name].append(export.despeckle(ink, args.min_speck))
            log.info("  drew frame %d/%d (%.1fs)", i, len(frames), time.perf_counter() - t)
            free_device_memory()
    finally:
        rora_log.setLevel(level)
    suffix = path.suffix.lower()
    sub_tag = "" if args.subject == "all" else f"_{args.subject}"
    written = 0
    for name in names:
        tag, settings = ids[name]
        stem = f"{path.stem}{sub_tag}__{tag}"
        gif = args.out / f"{stem}.gif"
        video.save_gif(inks[name], gif, args.color, durations, settings)
        outs = [gif]
        if suffix != ".gif":
            outs.insert(0, args.out / f"{stem}{suffix}")
            video.save_video(inks[name], outs[0], args.color, durations, args.antialias)
        written += len(outs)
        log.info("  %s: %s %dx%d, %d frame(s)", name, " + ".join(o.name for o in outs), w, h,
                 len(frames))
        middle = inks[name][len(frames) // 2]
        previews.append((stem, export.to_rgba(middle, args.color, args.antialias)))
    return written


@dataclass
class RefineJob:
    """A finished drawing waiting for the refine phase (ink and mask bit-packed to save memory)."""
    stem: str
    settings: str
    kinds: list
    shape: tuple
    ink: np.ndarray
    mask: np.ndarray

    def unpack(self, packed: np.ndarray) -> np.ndarray:
        return np.unpackbits(packed, count=self.shape[0] * self.shape[1]).reshape(self.shape) > 0


def refine_all(args, jobs: list, refine_opts, device: str, previews: list) -> int:
    """Phase 2: run every queued drawing through the editing model; returns files written."""
    written = 0
    finder = tattoo.get_finder("cpu")  # see tattoo.release_models()
    for i, job in enumerate(jobs, 1):
        log.info("[refine %d/%d] %s", i, len(jobs), job.stem)
        ink, mask = job.unpack(job.ink), job.unpack(job.mask)
        try:
            r = refine.refine(ink, mask, job.kinds, args.refine, finder, refine_opts, device)
        except refine.RefineError as e:
            log.error("  refine failed, keeping only the original: %s", e)
            continue
        if r.mode == "skip":
            log.info("  refine: score %.2f >= %.2f, skipped", r.score_before,
                     args.refine_threshold)
            continue
        rstem = f"{job.stem}_{refine_tag(r.mode, args)}"
        if args.debug:
            cv2.imwrite(str(args.out / f"{rstem}_debug_sent.png"), r.sent)
            cv2.imwrite(str(args.out / f"{rstem}_debug_raw.png"),
                        cv2.cvtColor(r.raw, cv2.COLOR_RGB2BGR))
        if r.ink is None:
            log.warning("  refine (%s) rejected, %s; keeping only the original", r.mode,
                        r.rejected)
            continue
        rout = export.despeckle(r.ink, args.min_speck)
        rout = export.crop_and_resize(rout, mask, args.pad, args.size)
        rsettings = json.dumps({**json.loads(job.settings), **r.settings}, sort_keys=True)
        rimg, rmsg = save_outputs(args, rout, rstem, rsettings)
        seam_note = (f", seams: {r.seams[0]} break(s) bridged, {r.seams[1]} line(s) removed"
                     if any(r.seams) else "")
        log.info("  refine (%s): %s, score %.2f -> %.2f%s (%.1fs)", r.mode, rmsg,
                 r.score_before, r.score_after, seam_note, r.seconds)
        previews.append((rstem, rimg))
        written += 1
    return written


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_styles:
        list_styles()
        return 0

    groups = {"all": list(styles.STYLES), "tattoo": tattoo.NAMES}
    names = [n for s in args.style for n in groups.get(s, [s])]
    unknown = [n for n in names if n not in styles.STYLES]
    if unknown:
        sys.exit(f"Unknown style(s): {unknown}. Use --list-styles.")

    setup_logging(args.out)
    overrides = parse_overrides(args)
    files = collect_images(args.inputs)
    if args.sample:
        files = sample_images(files, args.sample, args.sample_seed)
    log.info("%d photo(s), style(s): %s, overrides: %s", len(files), names, overrides or "none")
    ids = {n: settings_id(n, overrides, args) for n in names}
    for n, (tag, text) in ids.items():
        log.info("  %s -> <photo>__%s.png  %s", n, tag, text)
    anim_ids = {n: settings_id(n, overrides, args, animated=True) for n in names}
    anims = [f for f in files if video.is_animation(f)]
    if anims:
        for n, (tag, text) in anim_ids.items():
            log.info("  %s -> <clip>__%s.gif (+ .mp4|.mov for videos)  %s", n, tag, text)
        if args.refine != "off" or args.svg or args.debug:
            log.warning("  --refine, --svg and --debug are skipped for GIFs and videos")
    used_keys = {k for n in names for k in styles.STYLES[n].defaults} | {"invert"}
    ignored = sorted({s.partition("=")[0] for s in args.set} - used_keys)
    if ignored:
        log.warning("  --set %s not used by style(s) %s", ", ".join(ignored), names)
    if not files:
        return 1

    refine_opts = None
    if args.refine != "off":
        try:
            backend = refine.check_ready(args.refine_backend, args.refine_model)
            warning = refine.check_memory(args.refine_model, backend, args.refine_quantize,
                                          args.refine_force)
        except refine.RefineError as e:
            log.error("%s", e)
            return 2
        if warning:
            log.warning("Refine memory: %s", warning)
        refine_opts = refine.Options(backend, args.refine_prompt, args.refine_seed,
                                     args.refine_threshold, args.refine_quantize,
                                     args.refine_model)
        log.info("Refine: %s with %s via %s", args.refine,
                 refine.MODELS[args.refine_model].name, backend)

    seg = Segmenter(args.device, args.detector, args.segmenter, args.min_conf)
    device = seg.device
    previews, written, skipped = [], 0, []
    refine_jobs = [] if refine_opts else None
    start = time.perf_counter()

    for i, path in enumerate(files, 1):
        log.info("[%d/%d] %s", i, len(files), path.name)
        try:
            if video.is_animation(path):
                n = draw_animation(args, path, seg, names, anim_ids, overrides, previews)
                written += n
                if not n:
                    skipped.append(path.name)
                continue  # draw_animation frees device memory after every frame

            rgb = load_image(path, args.work_size)
            subjects = seg.find_subjects(rgb, args.subject, args.subject_conf)
            if not subjects:
                log.warning("  no %s found, skipping", args.subject
                            if args.subject in COCO_KINDS.values() else "person, cat or dog")
                skipped.append(path.name)
                continue

            for sub_tag, group in subject_groups(subjects, args.subject):
                log.info("  drawing %s%s", describe([s.kind for s in group]),
                         f" -> {path.stem}{sub_tag}" if sub_tag else "")
                written += draw_subjects(args, path.stem + sub_tag, rgb, group, names, ids,
                                         overrides, refine_jobs, previews)
        except Exception as e:  # keep the batch going; one bad photo shouldn't stop the run
            log.exception("  failed: %s", e)
            skipped.append(path.name)
        free_device_memory()  # otherwise torch's MPS cache grows with every photo (~1-2 GB each)

    if refine_jobs:
        # the image-editing model needs the memory the drawing models were holding
        log.info("Refine phase: freeing the drawing models, then %d drawing(s) through %s",
                 len(refine_jobs), refine.MODELS[refine_opts.model].name)
        del seg
        tattoo.release_models()
        free_device_memory()
        written += refine_all(args, refine_jobs, refine_opts, device, previews)

    if args.preview:
        export.save_preview(previews, args.out / "_preview.png", args.color)
        log.info("Preview: %s", args.out / "_preview.png")
    log.info("Done: %d file(s) written, %d photo(s) skipped in %.0fs", written, len(skipped),
             time.perf_counter() - start)
    if skipped:
        log.info("Skipped: %s", ", ".join(skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
