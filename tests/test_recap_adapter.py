import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from lingbotvla.recap.adapter import (
    apply_recap_velocity_lora,
    build_recap_condition_embedding,
    load_counterfactual_decision_map,
    load_recap_adapter_registry,
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

    def test_loads_relative_task_adapter_registry_and_checks_hash(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "bell.safetensors"
            artifact.write_bytes(b"adapter")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "tasks": {
                            "click_bell": {
                                "path": artifact.name,
                                "sha256": digest,
                                "condition_start_decision": 1,
                                "condition_decisions": 1,
                            }
                        },
                    }
                )
            )
            loaded = load_recap_adapter_registry(registry)
            self.assertEqual(
                loaded["tasks"]["click_bell"]["path"], artifact.resolve()
            )
            self.assertEqual(loaded["tasks"]["click_bell"]["sha256"], digest)
            self.assertEqual(
                loaded["tasks"]["click_bell"]["condition_start_decision"], 1
            )
            self.assertEqual(loaded["tasks"]["click_bell"]["condition_decisions"], 1)

            artifact.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_recap_adapter_registry(registry)

    def test_loads_and_validates_counterfactual_decision_map(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.json"
            path.write_text(
                json.dumps(
                    {"schema_version": 1, "tasks": {"click_bell": 1}}
                )
            )
            loaded = load_counterfactual_decision_map(path)
            self.assertEqual(loaded["tasks"], {"click_bell": 1})
            path.write_text(
                json.dumps(
                    {"schema_version": 1, "tasks": {"click_bell": -1}}
                )
            )
            with self.assertRaisesRegex(ValueError, "non-negative"):
                load_counterfactual_decision_map(path)

    def test_rejects_invalid_adapter_registry(self):
        with TemporaryDirectory() as directory:
            registry = Path(directory) / "registry.json"
            registry.write_text(json.dumps({"schema_version": 2, "tasks": {}}))
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_recap_adapter_registry(registry)

    def test_rejects_invalid_registry_condition_window(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "adapter.safetensors"
            artifact.write_bytes(b"adapter")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "tasks": {
                            "task": {
                                "path": artifact.name,
                                "sha256": digest,
                                "condition_start_decision": -1,
                            }
                        },
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "non-negative integer"):
                load_recap_adapter_registry(registry)

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
