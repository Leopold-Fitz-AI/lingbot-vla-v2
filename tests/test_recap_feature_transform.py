import unittest


try:
    from lingbotvla.data.vla_data.utils import FeatureTransform
except ModuleNotFoundError as exc:
    FeatureTransform = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(FeatureTransform is None, f"optional training dependency unavailable: {IMPORT_ERROR}")
class RecapFeatureTransformPromptTest(unittest.TestCase):
    @staticmethod
    def make_transform(
        *, enabled=True, dropout=0.0, missing="positive", prompt_enabled=True
    ):
        transform = FeatureTransform.__new__(FeatureTransform)
        transform.recap_enabled = enabled
        transform.recap_indicator_key = "recap_label"
        transform.recap_missing_condition = missing
        transform.recap_condition_dropout = dropout
        transform.recap_condition_name = "Advantage"
        transform.recap_prompt_enabled = prompt_enabled
        transform.recap_adapter_enabled = False
        return transform

    def test_disabled_path_preserves_original_task(self):
        transform = self.make_transform(enabled=False)
        self.assertEqual(
            transform.build_task_prompt({"task": "pick cup", "recap_label": 0}),
            "pick cup",
        )

    def test_training_and_deployment_share_condition_format(self):
        transform = self.make_transform()
        item = {"task": "pick cup", "recap_label": 1}
        expected = "Advantage: positive.\nTask: pick cup"
        self.assertEqual(transform.build_task_prompt(item), expected)
        self.assertEqual(transform.build_task_prompt(item, policy_eval=True), expected)

    def test_legacy_demonstration_defaults_to_positive(self):
        transform = self.make_transform(missing="positive")
        self.assertEqual(
            transform.build_task_prompt({"task": "pick cup"}, policy_eval=True),
            "Advantage: positive.\nTask: pick cup",
        )

    def test_adapter_mode_can_leave_task_prompt_unchanged(self):
        transform = self.make_transform(prompt_enabled=False)
        self.assertEqual(
            transform.build_task_prompt({"task": "pick cup", "recap_label": 1}),
            "pick cup",
        )

    def test_dropout_builds_true_unconditional_prompt(self):
        transform = self.make_transform(dropout=1.0)
        self.assertEqual(
            transform.build_task_prompt({"task": "pick cup", "recap_label": 1}),
            "pick cup",
        )
        self.assertEqual(
            transform.build_task_prompt(
                {"task": "pick cup", "recap_label": 1},
                policy_eval=True,
            ),
            "Advantage: positive.\nTask: pick cup",
        )


if __name__ == "__main__":
    unittest.main()
