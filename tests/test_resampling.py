import unittest

import numpy as np

from smolvla_cf.export import _action_chunks, resample_trajectory
from smolvla_cf.evaluate import resample_policy_targets_bspline_for_sim


class ResamplingTest(unittest.TestCase):
    def setUp(self):
        count = 6
        timestamp = np.arange(count, dtype=np.float64) / 50
        state = np.zeros((count, 6), dtype=np.float32)
        action = np.zeros((count, 6), dtype=np.float32)
        state[:, :5] = timestamp[:, None] * np.arange(1, 6)
        action[:, :5] = 2 * timestamp[:, None] * np.arange(1, 6)
        state[:, 5] = np.arange(count) + 10
        action[:, 5] = np.arange(count) + 20
        images = np.broadcast_to(
            np.arange(count, dtype=np.uint8)[:, None, None, None], (count, 2, 2, 3)
        ).copy()
        self.source = {
            "timestamp": timestamp,
            "observation.state": state,
            "action": action,
            "observation.images.overhead": images,
            "observation.images.wrist": images,
            "phase": np.arange(count, dtype=np.int8),
            "terminal": np.asarray([False] * (count - 1) + [True]),
        }

    def test_exact_30_hz_timeline_and_modalities(self):
        result = resample_trajectory(self.source, 30)
        target_t = np.arange(4) / 30
        np.testing.assert_allclose(result["timestamp"], target_t, atol=1e-7)
        np.testing.assert_allclose(
            result["observation.state"][:, :5], target_t[:, None] * np.arange(1, 6), atol=1e-6
        )
        np.testing.assert_allclose(
            result["action"][:, :5], 2 * target_t[:, None] * np.arange(1, 6), atol=1e-6
        )
        np.testing.assert_array_equal(result["source_nearest_index"], [0, 2, 3, 5])
        np.testing.assert_array_equal(result["observation.state"][:, 5], [10, 12, 13, 15])
        np.testing.assert_array_equal(result["action"][:, 5], [20, 21, 23, 25])
        np.testing.assert_array_equal(result["terminal"], [False, False, False, True])

    def test_action_chunks_repeat_terminal_and_mark_padding(self):
        result = resample_trajectory(self.source, 30)
        chunks, is_pad = _action_chunks(
            result["action"], result["observation.state"], 4, result["terminal_action"]
        )
        np.testing.assert_array_equal(is_pad[-1], [False, True, True, True])
        np.testing.assert_allclose(chunks[-1, 1:], np.repeat(chunks[-1:, 0], 3, axis=0))
        decoded_arm = chunks[0, :, :5] + result["observation.state"][0, :5]
        np.testing.assert_allclose(decoded_arm, result["action"][:, :5], atol=1e-6)

    def test_padding_uses_actual_final_50_hz_target(self):
        short = {key: value[:3] for key, value in self.source.items()}
        short["terminal"] = np.asarray([False, False, True])
        result = resample_trajectory(short, 30)
        chunks, is_pad = _action_chunks(
            result["action"], result["observation.state"], 3, result["terminal_action"]
        )
        self.assertTrue(is_pad[-1, 1:].all())
        decoded = chunks[-1] + np.r_[result["observation.state"][-1, :5], 0]
        # Gripper is absolute, so undo the helper addition for that channel.
        decoded[:, 5] = chunks[-1, :, 5]
        np.testing.assert_allclose(decoded[1:], np.repeat(self.source["action"][2:3], 2, axis=0))

    def test_bspline_100hz_is_joined_to_state_and_sampled_at_50hz(self):
        current = np.asarray([10, 20, 30, 40, 50, 0], dtype=np.float32)
        targets = np.stack(
            [current + np.asarray([i, 2 * i, 3 * i, 4 * i, 5 * i, 100], dtype=np.float32)
             for i in range(1, 13)]
        )
        result = resample_policy_targets_bspline_for_sim(targets, current, 30, 100)
        self.assertEqual(result.shape, (20, 6))
        np.testing.assert_allclose(result[0, :5], current[:5], atol=1e-6)
        self.assertEqual(result[0, 5], current[5])
        self.assertEqual(result[1, 5], current[5])
        self.assertEqual(result[2, 5], targets[0, 5])
        for channel in range(5):
            self.assertGreaterEqual(result[:, channel].min(), current[channel])
            self.assertLessEqual(result[:, channel].max(), targets[-1, channel])


if __name__ == "__main__":
    unittest.main()
