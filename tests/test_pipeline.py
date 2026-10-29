"""High-level pipeline tests: each style once (with varied options), and every input file type.

Run from the repo root:
    env/bin/python -m unittest -v
    env/bin/python -m unittest -v tests.test_pipeline.StyleTests.test_edges   # just one
Set RORA_KEEP_TEST_OUTPUT=1 to keep the generated inputs and outputs (the path is printed).
"""
import contextlib
import io
import json
import logging
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pillow_heif
from PIL import Image

import lineart
from rora import export
from rora.segment import load_image

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "img" / "in" / "1.jpg"
STILL_EXTS = (".jpg", ".jpeg", ".png", ".heic", ".webp", ".tif", ".tiff")
PHOTO_SIZE, ANIM_SIZE = 600, 400
# small working sizes and 2 frames per clip keep each test to seconds
FAST = ["--work-size", "800", "--size", str(PHOTO_SIZE), "--anim-size", str(ANIM_SIZE),
        "--max-frames", "2"]
CLIP_FRAMES, CLIP_MS = 6, 100


def make_inputs(folder: Path) -> dict[str, Path]:
    """The demo photo in each still format, and a short pan across it as GIF, MP4 and MOV."""
    rgb = load_image(DEMO, 900)
    paths = {ext: folder / f"photo{ext}" for ext in (*STILL_EXTS, ".gif", ".mp4", ".mov")}
    for ext in STILL_EXTS:
        if ext == ".heic":
            pillow_heif.from_pillow(Image.fromarray(rgb)).save(paths[ext])
        else:
            Image.fromarray(rgb).save(paths[ext])

    h, w = rgb.shape[:2]
    width = int(w * 0.9) // 2 * 2
    step = (w - width) // (CLIP_FRAMES - 1)
    frames = [np.ascontiguousarray(rgb[:, i * step:i * step + width]) for i in range(CLIP_FRAMES)]
    gif = [Image.fromarray(f) for f in frames]
    gif[0].save(paths[".gif"], save_all=True, append_images=gif[1:], duration=CLIP_MS, loop=0)
    for ext in (".mp4", ".mov"):
        writer = cv2.VideoWriter(str(paths[ext]), cv2.VideoWriter_fourcc(*"mp4v"),
                                 1000 / CLIP_MS, (width, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
    return paths


class PipelineTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DEMO.exists():
            raise unittest.SkipTest(f"demo photo {DEMO} is missing")
        cls.tmp = Path(tempfile.mkdtemp(prefix="rora_tests_"))
        cls.inputs = make_inputs(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        if os.environ.get("RORA_KEEP_TEST_OUTPUT"):
            print(f"\nTest files kept in {cls.tmp}")
        else:
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_pipeline(self, style: str, ext: str, *opts: str) -> Path:
        """Run lineart.py on the input of type `ext`; returns the output folder."""
        out = self.tmp / f"out_{style}"
        root = logging.getLogger()
        before = list(root.handlers)
        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                code = lineart.main([str(self.inputs[ext]), "-o", str(out), "-s", style,
                                     *FAST, *opts])
        finally:
            for h in root.handlers[:]:
                if h not in before:  # main() adds a console and a run.log handler per call
                    root.removeHandler(h)
                    h.close()
        text = log.getvalue()
        self.assertEqual(code, 0, text[-3000:])
        self.assertIn(" 0 photo(s) skipped", text, text[-3000:])  # main() logs failures, carries on
        return out

    def check_still(self, style: str, ext: str, *opts: str):
        out = self.run_pipeline(style, ext, *opts)
        color = "black" if "black" in opts else "white"
        pngs = sorted(out.glob(f"*__{style}_*.png"))
        self.assertTrue(pngs, f"no {style} PNG in {out}")
        for png in pngs:
            with Image.open(png) as im:
                self.assertEqual(im.mode, "RGBA")
                self.assertAlmostEqual(max(im.size), PHOTO_SIZE, delta=1)
                self.assertEqual(json.loads(im.text["dog_lineart_settings"])["style"], style)
                rgba = np.asarray(im)
            ink = rgba[..., 3] > 0
            self.assertTrue(ink.any(), f"{png.name} has no ink")
            self.assertTrue((rgba[ink][:, :3] == export.COLORS[color]).all())
            self.assertEqual(png.with_suffix(".svg").exists(), "--svg" in opts)
            self.assertEqual(png.with_suffix(".webp").exists(), "--sticker" in opts)
            if "--sticker" in opts:
                with Image.open(png.with_suffix(".webp")) as im:
                    self.assertEqual((im.format, im.size), ("WEBP", (512, 512)))
        self.assertEqual((out / "_preview.png").exists(), "--preview" in opts)
        self.assertTrue((out / "run.log").exists())

    def check_clip(self, style: str, ext: str, *opts: str, frames: int = 2):
        out = self.run_pipeline(style, ext, *opts)
        color = "black" if "black" in opts else "white"
        gifs = sorted(out.glob(f"*__{style}_*.gif"))
        self.assertEqual(len(gifs), 1, f"expected one {style} .gif in {out}")
        with Image.open(gifs[0]) as im:
            self.assertEqual(im.n_frames, frames)
            if "--square" in opts:
                self.assertEqual(im.size, (ANIM_SIZE, ANIM_SIZE))
            self.assertEqual(max(im.size), ANIM_SIZE)
            opaque = "--opaque-background" in opts
            self.assertEqual("transparency" in im.info, not opaque)
            self.assertEqual(json.loads(im.info["comment"])["style"], style)
            bg = np.array(export.BACKGROUNDS[color], np.int16)
            inked = haloed = 0
            for i in range(im.n_frames):
                im.seek(i)
                rgba = np.asarray(im.convert("RGBA"))
                if opaque:
                    self.assertTrue((rgba[..., 3] == 255).all())
                    inked += bool((np.abs(rgba[..., :3] - bg).max(axis=2) > 100).any())
                else:
                    inked += bool((rgba[..., 3] > 0).any())
                    haloed += bool(((rgba[..., 3] > 0) & (rgba[..., :3] == bg).all(axis=2)).any())
            if not opaque:
                self.assertEqual(haloed > 0, "--halo" in opts)
        self.assertGreater(inked, 0, f"{gifs[0].name} has no ink in any frame")
        video_exts = {ext, ".mp4"} if "--mp4" in opts else {ext}
        if ".mp4" not in video_exts:
            self.assertEqual(sorted(out.glob(f"*__{style}_*.mp4")), [])
        for vext in sorted(video_exts - {".gif"}):
            files = sorted(out.glob(f"*__{style}_*{vext}"))
            self.assertEqual(len(files), 1, f"expected one {style} {vext} in {out}")
            cap = cv2.VideoCapture(str(files[0]))
            size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.assertEqual(max(size), ANIM_SIZE)
            if "--square" in opts:
                self.assertEqual(size, (ANIM_SIZE, ANIM_SIZE))
            bg = np.array(export.BACKGROUNDS[color][::-1], np.int16)
            n = inked = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                n += 1
                inked += bool((np.abs(bgr.astype(np.int16) - bg).max(axis=2) > 100).any())
            cap.release()
            self.assertEqual(n, frames)
            self.assertGreater(inked, 0, f"{files[0].name} has no ink in any frame")
        stickers = sorted(out.glob(f"*__{style}_*.webp"))
        if "--sticker" not in opts:
            self.assertEqual(stickers, [])
        else:
            self.assertEqual(len(stickers), 1, f"expected one {style} .webp sticker in {out}")
            with Image.open(stickers[0]) as im:
                self.assertEqual((im.format, im.size), ("WEBP", (512, 512)))
                self.assertEqual(im.n_frames, frames)
                self.assertTrue((np.asarray(im.convert("RGBA"))[..., 3] == 0).any())
        self.assertEqual((out / "_preview.png").exists(), "--preview" in opts)


class StyleTests(PipelineTestCase):
    """One test per style; together they cover every input file type."""

    # line / texture styles
    def test_stencil(self):
        self.check_still("stencil", ".jpg", "--svg")

    def test_ink(self):
        self.check_still("ink", ".jpeg", "--color", "black", "--weight", "3")

    def test_edges(self):
        self.check_still("edges", ".png", "--detail", "0.2", "--antialias", "--preview")

    def test_outline(self):
        self.check_still("outline", ".webp", "--outline", "3", "--subject", "auto")

    # photo-real tonal styles
    def test_halftone(self):
        self.check_still("halftone", ".tif", "--invert", "--set", "cell=10")

    def test_stipple(self):
        self.check_still("stipple", ".tiff", "--subject", "dog", "--min-speck", "0")

    def test_engraving(self):
        # 100 ms frames at 5 fps: every other frame is kept
        self.check_clip("engraving", ".gif", "--fps", "5", "--max-frames", "3", "--halo", "--mp4",
                        frames=3)

    # cartoony styles
    def test_cartoon(self):
        self.check_clip("cartoon", ".mp4", "--color", "black", "--antialias", "--sticker",
                        "--square")

    def test_comic(self):
        self.check_clip("comic", ".mov", "--subject", "dog", "--fps", "0", "--opaque-background",
                        "--mp4")

    # dramatic styles
    def test_woodcut(self):
        self.check_still("woodcut", ".heic", "--detail", "0.8", "--pad", "0.1")

    def test_noir(self):
        self.check_still("noir", ".jpg", "--style-defaults", "--preview")

    def test_sketch(self):
        self.check_clip("sketch", ".gif", "--color", "black", "--outline", "2", "--preview",
                        "--no-sticker")

    # tattoo looks (slower: several models per subject)
    def test_fineline(self):
        self.check_still("fineline", ".jpg", "--svg", "--set", "almond_eyes=1")

    def test_bold(self):
        self.check_still("bold", ".png", "--color", "black", "--sticker")

    def test_brush(self):
        self.check_still("brush", ".webp", "--detail", "0")

    def test_sketchy(self):
        self.check_clip("sketchy", ".mp4", "--max-frames", "1", frames=1)

    def test_dotwork(self):
        self.check_still("dotwork", ".tiff", "--subject", "auto", "--antialias")

    def test_minimal(self):
        self.check_still("minimal", ".jpeg", "--set", "pupils=1", "--outline", "2")


class FileTypeTests(PipelineTestCase):
    def test_every_style_has_a_test(self):
        tested = {name[5:] for name in dir(StyleTests) if name.startswith("test_")}
        self.assertEqual(tested, set(lineart.styles.STYLES))

    def test_every_file_type_is_picked_up(self):
        found = {p.suffix for p in lineart.collect_images([self.tmp])}
        self.assertEqual(found, {*STILL_EXTS, ".gif", ".mp4", ".mov"})

    def test_inputs_are_required(self):
        with self.assertRaises(SystemExit) as e:
            lineart.main([])
        self.assertIn("give one or more", str(e.exception.code))


if __name__ == "__main__":
    unittest.main()
