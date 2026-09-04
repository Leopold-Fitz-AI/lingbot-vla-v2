import unittest

import torch

from lingbotvla.recap.adapter import (
    apply_recap_velocity_lora,
    build_recap_condition_embedding,
)


class RecapAdapterTest(unittest.TestCase):
    def test_null_is_exact_zero_and_conditions_select_distinct_rows(self):
        parameters = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            requires_grad=True,
        )
        result = build_recap_condition_embedding(
            parameters,
            torch.tensor([-1, 0, 1]),
            batch_size=3,
            device="cpu",
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(result[0], torch.zeros(2)))
        self.assertTrue(torch.equal(result[1], parameters[0]))
        self.assertTrue(torch.equal(result[2], parameters[1]))

    def test_null_has_graph_edge_but_zero_gradient(self):
        parameters = torch.randn(2, 4, requires_grad=True)
        null = build_recap_condition_embedding(
            parameters,
            None,
            batch_size=2,
            device="cpu",
            dtype=torch.float32,
        )
        null.sum().backward()
        self.assertIsNotNone(parameters.grad)
        self.assertTrue(torch.equal(parameters.grad, torch.zeros_like(parameters)))

    def test_velocity_lora_is_null_preserving_and_condition_specific(self):
        hidden = torch.tensor([[[1.0, 2.0]], [[1.0, 2.0]], [[1.0, 2.0]]])
        # Negative selects hidden[0], positive selects hidden[1].
        lora_a = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        lora_b = torch.tensor([[[2.0]], [[3.0]]])
        residual = apply_recap_velocity_lora(
            hidden,
            [-1, 0, 1],
            lora_a,
            lora_b,
        )
        self.assertTrue(torch.equal(residual[0], torch.zeros_like(residual[0])))
        self.assertTrue(torch.equal(residual[1], torch.tensor([[2.0]])))
        self.assertTrue(torch.equal(residual[2], torch.tensor([[6.0]])))

    def test_signed_velocity_axis_is_exactly_antisymmetric(self):
        hidden = torch.tensor([[[1.0, 2.0]], [[1.0, 2.0]], [[1.0, 2.0]]])
        lora_a = torch.tensor([[[9.0, 9.0]], [[0.0, 1.0]]])
        lora_b = torch.tensor([[[9.0]], [[3.0]]])
        residual = apply_recap_velocity_lora(
            hidden,
            [-1, 0, 1],
            lora_a,
            lora_b,
            signed_axis=True,
        )
        self.assertTrue(torch.equal(residual[0], torch.zeros_like(residual[0])))
        self.assertTrue(torch.equal(residual[1], -residual[2]))
        self.assertTrue(torch.equal(residual[2], torch.tensor([[6.0]])))

    def test_zero_up_projection_preserves_all_conditions_and_gets_gradient(self):
        hidden = torch.randn(2, 3, 4)
        lora_a = torch.randn(2, 2, 4, requires_grad=True)
        lora_b = torch.zeros(2, 5, 2, requires_grad=True)
        residual = apply_recap_velocity_lora(
            hidden,
            [0, 1],
            lora_a,
            lora_b,
        )
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        residual.sum().backward()
        self.assertGreater(torch.count_nonzero(lora_b.grad).item(), 0)

    def test_rejects_invalid_ids_and_shapes(self):
        parameters = torch.zeros(2, 4)
        with self.assertRaisesRegex(ValueError, "-1, 0, or 1"):
            build_recap_condition_embedding(
                parameters,
                [2],
                batch_size=1,
                device="cpu",
                dtype=torch.float32,
            )
        with self.assertRaisesRegex(ValueError, "one value per batch"):
            build_recap_condition_embedding(
                parameters,
                [0],
                batch_size=2,
                device="cpu",
                dtype=torch.float32,
            )


if __name__ == "__main__":
    unittest.main()
