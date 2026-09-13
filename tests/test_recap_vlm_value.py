import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from deploy.recap_rollout_recorder import RecapEpisodeRecorder
from lingbotvla.recap.value import (
    FrozenVLMValueModel,
    VLMPooledValueModel,
    categorical_value_loss,
    masked_mean_pool,
)
from lingbotvla.recap.vlm_pool import (
    encode_policy_prefix_hidden,
    load_vlm_embedding_cache,
    match_rollout_embeddings,
    write_vlm_embedding_cache,
)
from scripts.recap_cache_vlm_embeddings import cache_rollout_embeddings


class MaskedMeanPoolTest(unittest.TestCase):
    def test_mean_pool_ignores_padded_tokens(self):
        hidden = torch.tensor(
            [
                [[1.0, 0.0], [3.0, 3.0], [9.0, 9.0]],
                [[2.0, 2.0], [4.0, 4.0], [8.0, 8.0]],
            ]
        )
        mask = torch.tensor([[True, True, False], [True, False, False]])
        pooled = masked_mean_pool(hidden, mask)
        self.assertTrue(torch.allclose(pooled, torch.tensor([[2.0, 1.5], [2.0, 2.0]])))

    def test_two_dimensional_embeddings_pass_through(self):
        hidden = torch.randn(3, 5)
        self.assertTrue(torch.equal(masked_mean_pool(hidden), hidden))


class VLMPooledValueModelTest(unittest.TestCase):
    def test_head_reads_pooled_vlm_embeddings(self):
        model = VLMPooledValueModel(hidden_size=4, num_bins=5, value_min=-4, value_max=0)
        embeddings = torch.randn(2, 4, requires_grad=True)
        logits = model(embeddings)
        self.assertEqual(tuple(logits.shape), (2, 5))
        loss = categorical_value_loss(logits, torch.tensor([-2.0, 0.0]), model.value_support)
        loss.backward()
        self.assertIsNotNone(embeddings.grad)
        self.assertEqual(tuple(model.expected_value(embeddings.detach()).shape), (2,))

    def test_sequence_input_is_mask_pooled_before_the_head(self):
        model = VLMPooledValueModel(hidden_size=2, num_bins=3, value_min=-1, value_max=1)
        hidden = torch.zeros(1, 3, 2)
        hidden[0, 0] = 1.0
        hidden[0, 1] = 3.0
        hidden[0, 2] = 99.0
        mask = torch.tensor([[True, True, False]])
        pooled = masked_mean_pool(hidden, mask)
        self.assertTrue(torch.allclose(model(hidden, mask), model(pooled)))


class FrozenVLMValueModelTest(unittest.TestCase):
    def test_encoder_parameters_do_not_receive_gradients(self):
        encoder = nn.Linear(3, 4)
        value = VLMPooledValueModel(hidden_size=4, num_bins=3, value_min=-1, value_max=1)
        model = FrozenVLMValueModel(value, encoder=encoder)
        images = torch.randn(2, 3, requires_grad=True)
        logits = model(encoder_inputs=(images,))
        loss = categorical_value_loss(
            logits,
            torch.tensor([-1.0, 1.0]),
            model.value_support,
        )
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in value.parameters()))
        self.assertEqual(
            set(model.trainable_state_dict()),
            set(value.state_dict()),
        )


class FakePrefixFlow:
    hidden_size = 4

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, image_grid_thw=None):
        batch, tokens = lang_tokens.shape
        hidden = torch.zeros(batch, tokens, self.hidden_size, dtype=images.dtype)
        hidden[:, :, 0] = images.reshape(batch, -1).mean(dim=1, keepdim=True)
        hidden[:, :, 1] = lang_tokens.float().mean(dim=1, keepdim=True)
        att_masks = torch.zeros_like(lang_masks)
        position_ids = torch.zeros(batch, tokens, dtype=torch.long)
        return hidden, lang_masks, att_masks, position_ids, None, None

    class _Expert:
        def forward(self, **kwargs):
            prefix = kwargs["inputs_embeds"][0]
            return [prefix + 1.0, None], None, None

    qwenvl_with_expert = _Expert()


class EncodePolicyPrefixTest(unittest.TestCase):
    def test_prefix_encoder_runs_the_vlm_and_pools_language_mask(self):
        images = torch.ones(2, 1, 3, 2, 2)
        lang_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
        lang_masks = torch.tensor([[True, True, False], [True, False, False]])
        hidden, mask = encode_policy_prefix_hidden(
            FakePrefixFlow(),
            images,
            torch.ones(2, 1, dtype=torch.bool),
            lang_tokens,
            lang_masks,
        )
        self.assertEqual(tuple(hidden.shape), (2, 3, 4))
        self.assertTrue(torch.equal(mask, lang_masks))
        self.assertTrue(torch.allclose(hidden[:, :, 0], images.reshape(2, -1).mean(dim=1, keepdim=True) + 1.0))
        pooled = masked_mean_pool(hidden, mask)
        self.assertEqual(tuple(pooled.shape), (2, 4))


class EmbeddingCacheTest(unittest.TestCase):
    def test_cache_round_trip_matches_rollout_keys(self):
        embeddings = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        episode_ids = ["a", "a", "b"]
        decision_indices = [0, 1, 0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            write_vlm_embedding_cache(
                path,
                embeddings=embeddings,
                episode_ids=episode_ids,
                decision_indices=decision_indices,
            )
            cache = load_vlm_embedding_cache(path)
            self.assertEqual(cache["hidden_size"], 2)
            matched = match_rollout_embeddings(
                cache,
                [
                    type("Decision", (), {"episode_id": "b", "decision_index": 0})(),
                    type("Decision", (), {"episode_id": "a", "decision_index": 1})(),
                ],
            )
            self.assertTrue(torch.equal(matched, torch.tensor([[5.0, 6.0], [3.0, 4.0]])))

    def test_missing_rollout_key_is_fail_closed(self):
        cache = {
            "schema_version": 1,
            "hidden_size": 2,
            "episode_ids": ["a"],
            "decision_indices": [0],
            "embeddings": torch.ones(1, 2),
        }
        with self.assertRaisesRegex(KeyError, "a::1"):
            match_rollout_embeddings(
                cache,
                [type("Decision", (), {"episode_id": "a", "decision_index": 1})()],
            )

    def test_cache_rollout_embeddings_uses_injected_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RecapEpisodeRecorder(
                directory,
                episode_id="ep0",
                task="pick cup",
                metadata={"task_name": "pick_cup"},
            )
            for decision_index in range(2):
                action = np.zeros((2, 2), dtype=np.float32)
                recorder.record_step(
                    {"observation.state": np.array([decision_index], dtype=np.float32)},
                    action,
                    action,
                    terminated=decision_index == 1,
                )
            recorder.finalize(success=True, terminal_reason="success")

            def encode_fn(decision):
                hidden = torch.tensor(
                    [[[float(decision.decision_index), 1.0], [9.0, 9.0]]],
                    dtype=torch.float32,
                )
                mask = torch.tensor([[True, False]])
                return hidden, mask

            output = Path(directory) / "cache.pt"
            summary = cache_rollout_embeddings(directory, output, encode_fn, failure_penalty=10)
            self.assertEqual(summary["decisions"], 2)
            self.assertEqual(summary["hidden_size"], 2)
            cache = load_vlm_embedding_cache(output)
            self.assertTrue(torch.allclose(cache["embeddings"], torch.tensor([[0.0, 1.0], [1.0, 1.0]])))


if __name__ == "__main__":
    unittest.main()
