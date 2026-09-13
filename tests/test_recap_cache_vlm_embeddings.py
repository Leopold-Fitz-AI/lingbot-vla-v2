import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import scripts.recap_cache_vlm_embeddings as cache_script
from deploy.recap_rollout_recorder import RecapEpisodeRecorder
from lingbotvla.recap.decision_preprocess import (
    decision_to_prefix_inputs,
    load_decision_observation,
)
from lingbotvla.recap.policy_loader import training_config_path_for_checkpoint
from lingbotvla.recap.rollouts import load_rollout_decisions
from lingbotvla.recap.vlm_pool import (
    encode_policy_prefix_hidden,
    load_vlm_embedding_cache,
    match_rollout_embeddings,
)


IMAGE_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


def record_episode(root, episode_id, *, task="pick cup", success=True, cameras=IMAGE_KEYS):
    recorder = RecapEpisodeRecorder(
        root,
        episode_id=episode_id,
        task=task,
        metadata={"task_name": episode_id},
    )
    for decision_index, action_length in enumerate((2, 1)):
        action = np.zeros((action_length, 14), dtype=np.float32)
        observation = {
            "observation.state": np.full(14, decision_index, dtype=np.float32),
        }
        for camera_index, key in enumerate(cameras):
            observation[key] = np.full(
                (6, 10, 3),
                fill_value=10 * camera_index + decision_index,
                dtype=np.uint8,
            )
        recorder.record_step(
            observation,
            action,
            action,
            terminated=success and decision_index == 1,
            truncated=(not success and decision_index == 1),
        )
    recorder.finalize(
        success=success,
        terminal_reason="success" if success else "step_limit",
    )


class FakeFeatureTransform:
    """Minimal stand-in for ``FeatureTransform`` over recorded NPZ keys."""

    def __init__(self, image_keys=IMAGE_KEYS):
        self.org_features = {"images": list(image_keys), "states": ["observation.state"], "actions": []}
        self.recap_enabled = True
        self.recap_indicator_key = "recap_label"
        self.calls = []

    def apply(self, item, policy_eval=False):
        self.calls.append(
            {
                "policy_eval": policy_eval,
                "indicator": item.get(self.recap_indicator_key),
            }
        )
        images = torch.stack([item[key] for key in self.org_features["images"]], dim=0)
        lang_tokens = torch.arange(1, len(str(item["task"])) + 1, dtype=torch.long)
        return {
            "images": images,
            "img_masks": torch.ones(images.shape[0], dtype=torch.bool),
            "state": torch.as_tensor(item["observation.state"], dtype=torch.float32),
            "lang_tokens": lang_tokens,
            "lang_masks": torch.ones(lang_tokens.shape[0], dtype=torch.bool),
            "image_grid_thw": torch.tensor(
                [[1, images.shape[-2], images.shape[-1]]] * images.shape[0],
                dtype=torch.long,
            ),
        }


class FakePrefixFlow:
    """Test double for ``FlowMatchingV2`` (mirrors test_recap_vlm_value)."""

    hidden_size = 4

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, image_grid_thw=None):
        batch, tokens = lang_tokens.shape
        hidden = torch.zeros(batch, tokens, self.hidden_size, dtype=torch.float32)
        hidden[:, :, 0] = images.reshape(batch, -1).mean(dim=1, keepdim=True).float()
        hidden[:, :, 1] = lang_tokens.float().mean(dim=1, keepdim=True)
        hidden[:, :, 2] = img_masks.float().sum(dim=1, keepdim=True)
        att_masks = torch.zeros_like(lang_masks)
        position_ids = torch.zeros(batch, tokens, dtype=torch.long)
        return hidden, lang_masks, att_masks, position_ids, None, None

    class _Expert:
        def forward(self, **kwargs):
            prefix = kwargs["inputs_embeds"][0]
            return [prefix + 1.0, None], None, None

    qwenvl_with_expert = _Expert()


def fake_load_frozen_flow_model(checkpoint_path, robot_config, **kwargs):
    transform = FakeFeatureTransform()
    meta = {
        "policy_checkpoint": str(checkpoint_path),
        "robot_config": str(robot_config),
        "image_size": 8,
    }
    return FakePrefixFlow(), transform, meta


def expected_embedding(task, decision_index):
    image_mean = 10.0 + decision_index + 1.0
    token_mean = (len(task) + 1) / 2.0 + 1.0
    return torch.tensor([image_mean, token_mean, 4.0, 1.0])


class LoadDecisionObservationTest(unittest.TestCase):
    def test_rebuilds_raw_observation_with_task(self):
        with tempfile.TemporaryDirectory() as directory:
            record_episode(directory, "episode-0")
            decision = load_rollout_decisions(directory, failure_penalty=10)[1]
            observation = load_decision_observation(decision, IMAGE_KEYS)
            self.assertEqual(observation["task"], "pick cup")
            self.assertEqual(observation["observation.state"].dtype, np.float32)
            self.assertEqual(observation["observation.state"].shape, (14,))
            for index, key in enumerate(IMAGE_KEYS):
                image = observation[key]
                self.assertEqual(image.dtype, np.uint8)
                self.assertEqual(image.shape, (6, 10, 3))
                self.assertTrue((image == 10 * index + 1).all())

    def test_missing_camera_key_names_episode_and_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            record_episode(directory, "episode-0", cameras=IMAGE_KEYS[:2])
            decision = load_rollout_decisions(directory, failure_penalty=10)[0]
            with self.assertRaisesRegex(KeyError, "episode-0::0"):
                load_decision_observation(decision, IMAGE_KEYS)

    def test_missing_state_key_names_episode_and_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            record_episode(directory, "episode-0")
            decision = load_rollout_decisions(directory, failure_penalty=10)[0]
            with self.assertRaisesRegex(KeyError, "episode-0::0"):
                load_decision_observation(decision, IMAGE_KEYS, state_key="observation.missing")


class DecisionToPrefixInputsTest(unittest.TestCase):
    def test_null_condition_and_policy_eval_resize(self):
        observation = {
            key: np.full((6, 10, 3), fill_value=5 * index, dtype=np.uint8)
            for index, key in enumerate(IMAGE_KEYS)
        }
        observation["observation.state"] = np.zeros(14, dtype=np.float32)
        observation["task"] = "pick cup"
        transform = FakeFeatureTransform()
        inputs = decision_to_prefix_inputs(observation, transform, image_size=8)
        self.assertEqual(transform.calls, [{"policy_eval": True, "indicator": -1}])
        self.assertEqual(tuple(inputs["images"].shape), (3, 3, 8, 8))
        self.assertEqual(tuple(inputs["img_masks"].shape), (3,))
        self.assertEqual(tuple(inputs["state"].shape), (14,))
        self.assertEqual(tuple(inputs["image_grid_thw"].shape), (3, 3))
        self.assertEqual(
            set(inputs),
            {"images", "img_masks", "state", "lang_tokens", "lang_masks", "image_grid_thw"},
        )
        # Constant-value cameras stay constant under deployment-style resize.
        self.assertTrue(torch.allclose(inputs["images"][0], torch.full((3, 8, 8), 0.0)))
        self.assertTrue(torch.allclose(inputs["images"][1], torch.full((3, 8, 8), 5.0)))

    def test_missing_configured_camera_fails_closed(self):
        observation = {
            IMAGE_KEYS[0]: np.full((6, 10, 3), fill_value=1, dtype=np.uint8),
            "observation.state": np.zeros(14, dtype=np.float32),
            "task": "pick cup",
        }
        with self.assertRaisesRegex(ValueError, "cam_left_wrist"):
            decision_to_prefix_inputs(observation, FakeFeatureTransform(), image_size=8)


class CacheCliEndToEndTest(unittest.TestCase):
    def test_main_writes_verifiable_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rollout_dir = root / "rollouts"
            record_episode(rollout_dir, "episode-0", task="pick cup", success=True)
            record_episode(rollout_dir, "episode-1", task="place the red block", success=False)
            output = root / "vlm_cache.pt"
            with mock.patch.object(
                cache_script, "load_frozen_flow_model", fake_load_frozen_flow_model
            ):
                cache_script.main(
                    [
                        "--rollout-dir",
                        str(rollout_dir),
                        "--output",
                        str(output),
                        "--policy-checkpoint",
                        "/fake/checkpoint",
                        "--batch-size",
                        "2",
                        "--device",
                        "cpu",
                        "--failure-penalty",
                        "10",
                    ]
                )
            cache = load_vlm_embedding_cache(output)
            decisions = load_rollout_decisions(rollout_dir, failure_penalty=10)
            self.assertEqual(cache["episode_ids"], [item.episode_id for item in decisions])
            self.assertEqual(cache["decision_indices"], [item.decision_index for item in decisions])
            self.assertEqual(tuple(cache["embeddings"].shape), (4, FakePrefixFlow.hidden_size))
            matched = match_rollout_embeddings(cache, decisions)
            for row, decision in zip(matched, decisions):
                self.assertTrue(
                    torch.allclose(row, expected_embedding(decision.task, decision.decision_index))
                )
            self.assertEqual(cache["policy_checkpoint"], "/fake/checkpoint")
            self.assertEqual(cache["dtype"], "torch.float32")
            self.assertEqual(cache["num_decisions"], 4)
            self.assertIn("robot_config", cache)

    def test_main_without_policy_checkpoint_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            record_episode(directory, "episode-0")
            with self.assertRaisesRegex(ValueError, "--policy-checkpoint"):
                cache_script.main(
                    [
                        "--rollout-dir",
                        directory,
                        "--output",
                        str(Path(directory) / "cache.pt"),
                    ]
                )

    def test_preprocessing_failure_names_decision_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rollout_dir = root / "rollouts"
            record_episode(rollout_dir, "episode-bad", cameras=IMAGE_KEYS[:2])
            output = root / "vlm_cache.pt"
            with mock.patch.object(
                cache_script, "load_frozen_flow_model", fake_load_frozen_flow_model
            ):
                with self.assertRaisesRegex(ValueError, "episode-bad::0"):
                    cache_script.main(
                        [
                            "--rollout-dir",
                            str(rollout_dir),
                            "--output",
                            str(output),
                            "--policy-checkpoint",
                            "/fake/checkpoint",
                            "--device",
                            "cpu",
                        ]
                    )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()


class TrainingConfigPathTest(unittest.TestCase):
    def test_checkpoint_symlink_does_not_relocate_training_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            checkpoint_parent = run_dir / "checkpoints" / "global_step_50000"
            checkpoint_parent.mkdir(parents=True)
            (run_dir / "lingbotvla_cli.yaml").write_text("model: {}\n", encoding="utf-8")
            target = root / "verified-weights"
            target.mkdir()
            symlinked = checkpoint_parent / "hf_ckpt"
            symlinked.symlink_to(target, target_is_directory=True)

            resolved = training_config_path_for_checkpoint(symlinked)
            self.assertEqual(resolved, run_dir / "lingbotvla_cli.yaml")
            self.assertTrue(resolved.is_file())


class VisualPrecomputeCacheResetTest(unittest.TestCase):
    def test_encode_resets_stale_visual_precompute_cache(self):
        from types import SimpleNamespace

        flow = FakePrefixFlow()
        expert = FakePrefixFlow._Expert()
        expert.config = SimpleNamespace(precompute_grid_thw=True)
        expert.pos_embeds = object()
        expert.position_embeddings = object()
        expert.cu_seqlens = object()
        expert.visual_split_sizes = [64, 64, 64]
        expert.visual_max_seqlen = 64
        flow.qwenvl_with_expert = expert

        hidden, mask = encode_policy_prefix_hidden(
            flow,
            torch.zeros(1, 3, 2, 2),
            torch.ones(1, 3, dtype=torch.bool),
            torch.ones(1, 5, dtype=torch.long),
            torch.ones(1, 5, dtype=torch.bool),
        )
        self.assertEqual(tuple(hidden.shape), (1, 5, 4))
        self.assertEqual(tuple(mask.shape), (1, 5))
        for attr in (
            "pos_embeds",
            "position_embeddings",
            "cu_seqlens",
            "visual_split_sizes",
            "visual_max_seqlen",
        ):
            self.assertIsNone(getattr(expert, attr), attr)
