"""Tests for keeping subjects steady through an animation (no models needed).

    env/bin/python -m unittest -v tests.test_track
"""
import unittest

import numpy as np

from rora.segment import Subject
from rora.track import link, steady

H, W = 100, 200


def box_mask(box) -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[box[1]:box[3], box[0]:box[2]] = True
    return m


def subject(box, kind="dog") -> Subject:
    return Subject(box_mask(box), box, 0.9, kind)


def walking(n: int, missed=(), kind="dog", y=20):
    """A subject moving right 10 px a frame, missed by the detector in the `missed` frames."""
    return [[] if i in missed else [subject((10 + 10 * i, y, 50 + 10 * i, y + 40), kind)]
            for i in range(n)]


def segment_boxes(i, boxes):
    return [box_mask(b) for b in boxes]


class TrackTests(unittest.TestCase):
    def test_fills_frames_where_the_subject_was_missed(self):
        frames, calls = walking(8, missed={2, 3, 7}), []
        out, filled = steady(frames, lambda i, b: calls.append((i, b)) or segment_boxes(i, b))
        self.assertEqual(filled, {"dog": [2, 3, 7]})
        self.assertTrue(all(len(f) == 1 for f in out))
        # boxes between detections are interpolated; after the last one it's held
        self.assertEqual(calls, [(2, [(30, 20, 70, 60)]), (3, [(40, 20, 80, 60)]),
                                 (7, [(70, 20, 110, 60)])])
        self.assertEqual(out[2][0].score, 0.0)
        self.assertIs(out[0][0], frames[0][0])

    def test_rare_subjects_are_left_alone(self):
        frames = walking(6, missed={0, 2, 3, 5})
        out, filled = steady(frames, segment_boxes)
        self.assertEqual(filled, {})
        self.assertEqual([len(f) for f in out], [0, 1, 0, 0, 1, 0])

    def test_off(self):
        out, filled = steady(walking(6, missed={2}), segment_boxes, 1)
        self.assertEqual(filled, {})
        self.assertEqual(out[2], [])

    def test_bad_cut_outs_are_dropped(self):
        tiny = lambda i, boxes: [np.pad(np.ones((2, 2), bool), ((0, H - 2), (0, W - 2)))]
        _, filled = steady(walking(5, missed={2}), tiny)
        self.assertEqual(filled, {})
        empty = lambda i, boxes: [np.zeros((H, W), bool)]
        _, filled = steady(walking(5, missed={2}), empty)
        self.assertEqual(filled, {})

    def test_cut_out_on_another_subject_is_dropped(self):
        frames = walking(5, missed={2})
        frames[2] = [subject((0, 0, W, H), "person")]
        out, filled = steady(frames, segment_boxes)
        self.assertEqual(filled, {})
        self.assertEqual([s.kind for s in out[2]], ["person"])

    def test_two_subjects_tracked_apart(self):
        a = walking(6, missed={1}, y=0)
        b = walking(6, missed={4}, kind="cat", y=55)
        frames = [x + y for x, y in zip(a, b)]
        self.assertEqual(len(link(frames)), 2)
        out, filled = steady(frames, segment_boxes)
        self.assertEqual(filled, {"dog": [1], "cat": [4]})
        self.assertTrue(all(sorted(s.kind for s in f) == ["cat", "dog"] for f in out))

    def test_far_jump_after_a_gap_is_still_the_same_subject(self):
        frames = walking(6)
        frames[1:4] = [[], [], []]
        frames[4] = [subject((110, 20, 150, 60))]
        frames[5] = [subject((120, 20, 160, 60))]
        self.assertEqual(len(link(frames)), 1)


if __name__ == "__main__":
    unittest.main()
