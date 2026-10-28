"""Tests for the WhatsApp sticker output (no models needed).

    env/bin/python -m unittest -v tests.test_sticker
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image

from rora import sticker
from rora.export import COLORS


def ring(shape, center, r, width=3) -> np.ndarray:
    ink = np.zeros(shape, np.uint8)
    cv2.circle(ink, center, r, 1, width)
    return ink > 0


def noise(shape, seed) -> np.ndarray:
    return np.random.default_rng(seed).random(shape) > 0.5


def durations(path: Path) -> list[int]:
    with Image.open(path) as im:
        out = []
        for i in range(im.n_frames):
            im.seek(i)
            im.load()
            out.append(im.info["duration"])
    return out


class StickerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rora_sticker_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_still_is_a_512_transparent_webp_under_100_kb(self):
        path = self.tmp / "still.webp"
        note = sticker.save([ring((3000, 1800), (900, 1500), 700)], path, "white")
        self.assertEqual(note, "")
        self.assertLessEqual(path.stat().st_size, sticker.MAX_BYTES)
        with Image.open(path) as im:
            self.assertEqual(im.format, "WEBP")
            self.assertEqual(im.size, (512, 512))
            self.assertFalse(getattr(im, "is_animated", False))
            rgba = np.asarray(im.convert("RGBA"))
        self.assertEqual(rgba[0, 0, 3], 0)  # transparent background
        alpha = rgba[..., 3] > 0
        ys, xs = np.nonzero(alpha)
        # fitted inside the margin, centred, and filling it along the long side
        self.assertGreaterEqual(min(xs.min(), ys.min()), sticker.MARGIN)
        self.assertLessEqual(max(xs.max(), ys.max()), 511 - sticker.MARGIN)
        self.assertGreater(ys.max() - ys.min(), 512 - 2 * (sticker.MARGIN + sticker.HALO) - 8)
        # the lines are the ink colour, with a halo of the background colour around them
        solid = rgba[..., 3] == 255
        self.assertTrue((np.abs(rgba[solid][:, :3].astype(int) - COLORS["white"]) < 8).all(axis=1).any())
        self.assertTrue((rgba[solid][:, :3] < 60).all(axis=1).any())

    def test_animation_keeps_frames_and_timing(self):
        path = self.tmp / "anim.webp"
        inks = [ring((360, 640), (100 + 40 * i, 180), 60) for i in range(8)]
        note = sticker.save(inks, path, "black", [100] * 8)
        self.assertEqual(note, "")
        self.assertLessEqual(path.stat().st_size, sticker.MAX_ANIM_BYTES)
        with Image.open(path) as im:
            self.assertEqual(im.size, (512, 512))
            self.assertEqual(im.n_frames, 8)
        self.assertEqual(durations(path), [100] * 8)

    def test_long_clips_are_trimmed_to_10_seconds(self):
        path = self.tmp / "long.webp"
        inks = [ring((200, 200), (100, 100), 20 + i % 30) for i in range(150)]
        note = sticker.save(inks, path, "white", [100] * 150)
        self.assertIn("trimmed to 10 s", note)
        self.assertEqual(len(durations(path)), 100)

    def test_too_big_goes_lossy(self):
        path, ink = self.tmp / "lossy.webp", [ring((800, 800), (400, 400), 300)]
        lossless = len(sticker._encode(sticker.frames(ink, "white"), None, sticker.ENCODINGS[0]))
        with mock.patch.object(sticker, "MAX_BYTES", lossless - 1):
            note = sticker.save(ink, path, "white")
        self.assertIn("lossy", note)
        self.assertLess(path.stat().st_size, lossless)

    def test_still_too_big_drops_every_other_frame(self):
        path = self.tmp / "big.webp"
        inks = [noise((512, 512), i) for i in range(12)]
        note = sticker.save(inks, path, "white", [100] * 12)
        self.assertIn("of 12 frames", note)
        self.assertLessEqual(path.stat().st_size, sticker.MAX_ANIM_BYTES)
        kept = durations(path)
        self.assertLess(len(kept), 12)
        self.assertEqual(sum(kept), 1200)  # dropped frames' time goes to the kept ones


if __name__ == "__main__":
    unittest.main()
