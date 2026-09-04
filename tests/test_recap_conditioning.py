import unittest

from lingbotvla.recap.conditioning import (
    RecapCondition,
    format_recap_prompt,
    maybe_drop_recap_condition,
    normalize_recap_condition,
    recap_condition_id,
)


class NormalizeRecapConditionTest(unittest.TestCase):
    def test_normalizes_common_dataset_values(self):
        positive_values = (True, 1, 1.0, "positive", "high", "success")
        negative_values = (False, 0, 0.0, "negative", "low", "failure")
        for value in positive_values:
            with self.subTest(value=value):
                self.assertIs(normalize_recap_condition(value), RecapCondition.POSITIVE)
        for value in negative_values:
            with self.subTest(value=value):
                self.assertIs(normalize_recap_condition(value), RecapCondition.NEGATIVE)
        self.assertIs(normalize_recap_condition(-1), RecapCondition.NULL)

    def test_missing_condition_is_explicit(self):
        self.assertIs(
            normalize_recap_condition(None, missing="positive"),
            RecapCondition.POSITIVE,
        )
        self.assertIs(
            normalize_recap_condition(None, missing="null"),
            RecapCondition.NULL,
        )
        with self.assertRaisesRegex(ValueError, "Missing RECAP"):
            normalize_recap_condition(None, missing="error")

    def test_model_condition_ids(self):
        self.assertEqual(recap_condition_id("null"), -1)
        self.assertEqual(recap_condition_id("negative"), 0)
        self.assertEqual(recap_condition_id("positive"), 1)

    def test_rejects_non_binary_numbers(self):
        with self.assertRaisesRegex(ValueError, "-1, 0, or 1"):
            normalize_recap_condition(0.5)


class RecapPromptTest(unittest.TestCase):
    def test_condition_is_prepended(self):
        prompt = format_recap_prompt("pick up the cup", RecapCondition.POSITIVE)
        self.assertEqual(prompt, "Advantage: positive.\nTask: pick up the cup")

    def test_null_branch_preserves_original_prompt(self):
        self.assertEqual(
            format_recap_prompt("pick up the cup", RecapCondition.NULL),
            "pick up the cup",
        )

    def test_condition_dropout_is_deterministic_when_requested(self):
        self.assertIs(
            maybe_drop_recap_condition(
                RecapCondition.POSITIVE,
                0.25,
                random_value=0.1,
            ),
            RecapCondition.NULL,
        )
        self.assertIs(
            maybe_drop_recap_condition(
                RecapCondition.NEGATIVE,
                0.25,
                random_value=0.9,
            ),
            RecapCondition.NEGATIVE,
        )

    def test_invalid_dropout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            maybe_drop_recap_condition(RecapCondition.POSITIVE, 1.1)


if __name__ == "__main__":
    unittest.main()
