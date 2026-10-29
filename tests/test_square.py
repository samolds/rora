"""Tests for --square's sliding crop (no models needed).

    env/bin/python -m unittest -v tests.test_square
"""
import unittest

import numpy as np

from rora.segment import Subject
from rora.square import crop, crop_subject, offsets


class OffsetTests(unittest.TestCase):
    def test_centres_on_the_faces_along_the_long_side(self):
        # portrait 100x200: the square is 100 px and slides vertically (y)
        self.assertEqual(offsets([[(50, 120)], [(10, 60)]], (200, 100), 0), [70, 10])
        # landscape 200x100: it slides horizontally (x)
        self.assertEqual(offsets([[(120, 50)]], (100, 200), 0), [70])

    def test_several_faces_use_their_mean(self):
        self.assertEqual(offsets([[(50, 60), (50, 140)]], (200, 100), 0), [50])

    def test_stays_inside_the_frame(self):
        self.assertEqual(offsets([[(50, 5)], [(50, 195)]], (200, 100), 0), [0, 100])

    def test_fills_in_frames_without_faces(self):
        self.assertEqual(offsets([[(50, 60)], [], [], [(50, 120)]], (200, 100), 0),
                         [10, 30, 50, 70])
        self.assertEqual(offsets([[], [(50, 80)], []], (200, 100), 0), [30, 30, 30])

    def test_no_faces_or_already_square(self):
        self.assertEqual(offsets([[], []], (200, 100), 2), [50, 50])
        self.assertEqual(offsets([[(10, 10)]], (100, 100), 2), [0])

    def test_smoothing_softens_a_jump(self):
        jump = [[(50, 60)]] * 10 + [[(50, 140)]] * 10
        sharp, smooth = offsets(jump, (200, 100), 0), offsets(jump, (200, 100), 2)
        self.assertEqual(max(np.abs(np.diff(sharp))), 80)
        self.assertLess(max(np.abs(np.diff(smooth))), 20)
        self.assertEqual((smooth[0], smooth[-1]), (10, 90))


class CropTests(unittest.TestCase):
    def test_crops_frames_and_subjects(self):
        rgb = np.arange(200 * 100 * 3).reshape(200, 100, 3)
        self.assertTrue((crop(rgb, 30) == rgb[30:130]).all())
        self.assertTrue((crop(rgb.transpose(1, 0, 2), 30) == rgb.transpose(1, 0, 2)[:, 30:130])
                        .all())

        mask = np.zeros((200, 100), bool)
        mask[20:60, 10:40] = True
        s = crop_subject(Subject(mask, (10, 20, 40, 60), 0.9, "dog"), 30)
        self.assertEqual(s.mask.shape, (100, 100))
        self.assertEqual(s.box, (10, 0, 40, 30))
        self.assertEqual(int(s.mask.sum()), 30 * 30)


if __name__ == "__main__":
    unittest.main()
