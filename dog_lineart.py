#!/usr/bin/env python3
"""Photo of a dog -> transparent PNG line art (for printing on shirts)."""
import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import env  # noqa: F401  (must come before torch/ultralytics imports)
import cv2
import numpy as np

import export
import styles
import tattoo
from segment import DogSegmenter, load_image

log = logging.getLogger("dog_lineart")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tif", ".tiff"}

# Default --set values; any --set KEY=VALUE on the command line overrides these.
# They tune the tattoo looks (fineline, brush, ...) and are ignored by styles that don't use them.
DEFAULT_SETTINGS = {"max_strokes": 50, "head_strokes": 100, "form_lines": 100,
                    "min_votes": 2.5, "coverage": 0.1, "min_score": 0, "freedom": 0.4}

# command-line options that change the output image (others, like --svg, don't)
OUTPUT_OPTIONS = ("color", "size", "dpi", "pad", "min_speck", "outline", "antialias",
                  "work_size", "min_conf", "detector", "segmenter")


def settings_id(name: str, overrides: dict, args) -> tuple[str, str]:
    """(filename tag, full settings JSON) for one style. Only settings the style actually
    uses count, so the same effective settings always map to the same file name."""
    style = styles.STYLES[name]
    used = {k: v for k, v in {**style.defaults, **overrides}.items()
            if k in style.defaults or k == "invert"}
    full = {"style": name, **used, **{k: getattr(args, k) for k in OUTPUT_OPTIONS}}
    full = {k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
            for k, v in full.items()}  # so 1 and 1.0 hash the same
    text = json.dumps(full, sort_keys=True)
    digest = hashlib.sha1(text.encode()).hexdigest()[:6]
    return f"d{full['detail']:g}_{digest}", text


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="*", type=Path, default=[Path("img/in")],
                    help="image files or folders (default: img/in)")
    ap.add_argument("-o", "--out", type=Path, default=Path("img/out"))
    ap.add_argument("--list-styles", action="store_true", help="show styles and their settings")

    g = ap.add_argument_group("style")
    g.add_argument("-s", "--style", nargs="+", default=["edges"],
                   help="one or more style names, 'tattoo' (all tattoo looks) or 'all' (default: edges)")
    g.add_argument("--detail", type=float, default=0.5, help="0 (simple) .. 1 (busy), default 0.5")
    g.add_argument("--weight", type=int, help="line weight in px")
    g.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any style setting, e.g. --set threshold=0.45 (defaults: "
                        + " ".join(f"{k}={v}" for k, v in DEFAULT_SETTINGS.items()) + ")")
    g.add_argument("--invert", action="store_true",
                   help="flip tonal styles (ink where the dog is dark)")

    g = ap.add_argument_group("output")
    g.add_argument("--color", choices=["white", "black"], default="white", help="ink colour")
    g.add_argument("--size", type=int, default=3600, help="long side in px (0 = no resize)")
    g.add_argument("--dpi", type=int, default=300)
    g.add_argument("--pad", type=float, default=0.03, help="padding around dog, fraction")
    g.add_argument("--min-speck", type=int, default=12, help="drop specks smaller than this (px)")
    g.add_argument("--outline", type=int, default=0, help="add a silhouette outline this wide (px)")
    g.add_argument("--antialias", action="store_true", help="soft edges (hard by default)")
    g.add_argument("--svg", action="store_true", help="also write traced .svg files")
    g.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True,
                   help="write _preview.png contact sheet (default: on)")
    g.add_argument("--debug", action="store_true", help="save mask and cut-out per photo")

    g = ap.add_argument_group("detection")
    g.add_argument("--work-size", type=int, default=2500, help="processing size, long side px")
    g.add_argument("--min-conf", type=float, default=0.35, help="dog detection confidence")
    g.add_argument("--max-dogs", type=int, default=0, help="limit dogs per photo (0 = all)")
    g.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    g.add_argument("--detector", default="yolo11x.pt")
    g.add_argument("--segmenter", default="sam2.1_b.pt")
    return ap.parse_args(argv)


def parse_overrides(args) -> dict:
    out = dict(DEFAULT_SETTINGS)
    for item in args.set:
        key, _, value = item.partition("=")
        try:
            out[key] = float(value) if "." in value else int(value)
        except ValueError:
            out[key] = value
    if args.detail is not None:
        out["detail"] = args.detail
    if args.weight is not None:
        out["weight"] = args.weight
    if args.invert:
        out["invert"] = True
    return out


def collect_images(inputs: list[Path]) -> list[Path]:
    files = []
    for p in inputs:
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
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
    log.info("%d photo(s), style(s): %s, overrides: %s", len(files), names, overrides or "none")
    ids = {n: settings_id(n, overrides, args) for n in names}
    for n, (tag, text) in ids.items():
        log.info("  %s -> *_%s_%s.png  %s", n, n, tag, text)
    used_keys = {k for n in names for k in styles.STYLES[n].defaults} | {"invert"}
    ignored = sorted({s.partition("=")[0] for s in args.set} - used_keys)
    if ignored:
        log.warning("  --set %s not used by style(s) %s", ", ".join(ignored), names)
    if not files:
        return 1

    seg = DogSegmenter(args.device, args.detector, args.segmenter, args.min_conf)
    previews, written, skipped = [], 0, []
    start = time.perf_counter()

    for i, path in enumerate(files, 1):
        log.info("[%d/%d] %s", i, len(files), path.name)
        try:
            rgb = load_image(path, args.work_size)
            dogs = seg.find_dogs(rgb)
            if args.max_dogs:
                dogs = dogs[:args.max_dogs]
            if not dogs:
                log.warning("  no dog found, skipping")
                skipped.append(path.name)
                continue

            for d_i, dog in enumerate(dogs, 1):
                dog_tag = "" if d_i == 1 else f"_dog{d_i}"
                if args.debug:
                    cut = rgb.copy()
                    cut[~dog.mask] = 0
                    cv2.imwrite(str(args.out / f"{path.stem}{dog_tag}_debug_mask.png"),
                                dog.mask.astype(np.uint8) * 255)
                    cv2.imwrite(str(args.out / f"{path.stem}{dog_tag}_debug_cutout.jpg"),
                                cv2.cvtColor(cut, cv2.COLOR_RGB2BGR))

                for name in names:
                    t = time.perf_counter()
                    ink = styles.render(name, rgb, dog.mask, args.color == "white", overrides)
                    if args.outline:
                        ink |= styles.silhouette(dog.mask, args.outline)
                    ink = export.despeckle(ink, args.min_speck)
                    ink = export.crop_and_resize(ink, dog.mask, args.pad, args.size)

                    tag, settings = ids[name]
                    stem = f"{path.stem}{dog_tag}_{name}_{tag}"
                    img = export.save_png(ink, args.out / f"{stem}.png", args.color,
                                          args.antialias, args.dpi, settings)
                    msg = f"  {name}: {stem}.png {img.width}x{img.height}"
                    if args.svg:
                        n = export.save_svg(ink, args.out / f"{stem}.svg", args.color,
                                            settings=settings)
                        msg += f" + .svg ({n} paths)"
                    log.info("%s, ink %.0f%% (%.1fs)", msg, 100 * ink.mean(),
                             time.perf_counter() - t)
                    previews.append((stem, img))
                    written += 1
        except Exception as e:  # keep the batch going; one bad photo shouldn't stop the run
            log.exception("  failed: %s", e)
            skipped.append(path.name)

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
