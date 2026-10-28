"""Tests for drawing the face features (no models needed).

    env/bin/python -m unittest -v tests.test_facedraw
"""
import unittest

import numpy as np

from rora.facedraw import draw_face
from rora.features import Face, Feature

H = W = 400
SIZE = 300.0


def blob(box) -> Feature:
    """A feature whose SAM mask is an ellipse filling `box`."""
    x0, y0, x1, y1 = box
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    return Feature("f", box, 0.9, ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1)


def frontal_face() -> tuple[Face, np.ndarray]:
    face = Face(head=Feature("head", (80, 60, 320, 340), 0.9),
                eyes=[blob((130, 140, 180, 170)), blob((220, 140, 270, 170))],
                nose=blob((170, 200, 230, 240)), mouth=blob((150, 280, 250, 300)))
    return face, np.ones((H, W), bool)


class NoseTests(unittest.TestCase):
    def test_dogs_get_the_dog_nose(self):
        face, mask = frontal_face()
        strokes, fills = draw_face(face, np.zeros((H, W, 3), np.uint8), mask, SIZE, False,
                                   kind="dog")
        # eyes, closed nose outline, lip line and the line from nose to lip; two nostrils
        self.assertEqual(len(strokes), 5)
        self.assertEqual(len(fills), 2)
        solid_strokes, solid_fills = draw_face(face, np.zeros((H, W, 3), np.uint8), mask, SIZE,
                                               True, kind="dog")
        self.assertEqual(len(solid_fills), 1)

    def test_other_kinds_get_just_a_nose_line(self):
        face, mask = frontal_face()
        nose_bottom = 240
        for kind in ("person", "cat"):
            for solid in (False, True):
                with self.subTest(kind=kind, solid=solid):
                    strokes, fills = draw_face(face, np.zeros((H, W, 3), np.uint8), mask, SIZE,
                                               solid, kind=kind)
                    self.assertEqual(fills, [])  # no nostrils, no solid nose
                    self.assertEqual(len(strokes), 4)  # eyes, nose line, lip line
                    nose = strokes[2]
                    self.assertFalse(np.allclose(nose[0], nose[-1]), "nose line is not closed")
                    # it runs along the underside of the nose
                    self.assertGreater(nose[:, 1].mean(), (200 + nose_bottom) / 2)
                    self.assertLessEqual(nose[:, 1].max(), nose_bottom + 2)


if __name__ == "__main__":
    unittest.main()
