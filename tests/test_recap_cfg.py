import unittest

import torch

from lingbotvla.recap.cfg import (
    combine_cfg_velocities,
    combine_cfg_velocity_batch,
    duplicate_cfg_denoise_inputs,
)


class RecapCfgTest(unittest.TestCase):
    def test_scale_endpoints_and_extrapolation(self):
        positive = torch.tensor([[[3.0, 5.0]]])
        null = torch.tensor([[[1.0, 2.0]]])
        self.assertTrue(torch.equal(combine_cfg_velocities(positive, null, 0), null))
        self.assertTrue(torch.equal(combine_cfg_velocities(positive, null, 1), positive))
        self.assertTrue(
            torch.equal(
                combine_cfg_velocities(positive, null, 2),
                torch.tensor([[[5.0, 8.0]]]),
            )
        )

    def test_denoise_inputs_share_state_noise_and_time(self):
        state = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        x_t = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
        doubled_state, doubled_x_t, doubled_time = duplicate_cfg_denoise_inputs(
            state,
            x_t,
            torch.tensor(0.75),
        )
        self.assertTrue(torch.equal(doubled_state[:2], doubled_state[2:]))
        self.assertTrue(torch.equal(doubled_x_t[:2], doubled_x_t[2:]))
        self.assertTrue(torch.equal(doubled_time, torch.full((4,), 0.75)))

    def test_velocity_batch_order_is_positive_then_null(self):
        positive = torch.full((2, 3, 4), 4.0)
        null = torch.full((2, 3, 4), 1.0)
        velocity = torch.cat((positive, null), dim=0)
        combined = combine_cfg_velocity_batch(velocity, batch_size=2, scale=1.5)
        self.assertTrue(torch.equal(combined, torch.full((2, 3, 4), 5.5)))

    def test_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            combine_cfg_velocities(torch.zeros(1, 2), torch.zeros(2, 2), 1)
        with self.assertRaisesRegex(ValueError, "doubled"):
            combine_cfg_velocity_batch(torch.zeros(3, 2), batch_size=2, scale=1)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            combine_cfg_velocities(torch.zeros(1, 2), torch.zeros(1, 2), -1)


if __name__ == "__main__":
    unittest.main()
