import json

import pytest

from scripts.recap_compare_policy_variants import compare_variants


def _rows(root, outcomes):
    for seed, success in enumerate(outcomes):
        folder = root / str(seed)
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_text(json.dumps({
            "task": f"instruction-{seed}", "seed": seed, "success": success,
            "metadata": {"task_name": "bell", "policy_seed": "900", "continuation_policy_seed": "200",
                         "counterfactual_policy_decision": 1, "common_noise_per_episode": "true",
                         "task_config": "clean", "step_limit": 400},
        }))


def test_three_state_null_ceiling_is_inconclusive_not_failed_learning(tmp_path):
    variants = {name: tmp_path / name for name in ("null", "v22", "v2")}
    _rows(variants["null"], [True, True, True])
    _rows(variants["v22"], [True, True, True])
    _rows(variants["v2"], [True, True, False])
    result = compare_variants(variants, reference="null", seed_map={"schema_version": 1, "tasks": {"bell": [0, 1, 2]}})
    task = result["tasks"]["bell"]
    assert task["success"] == {"null": 3, "v22": 3, "v2": 2}
    assert task["comparisons_vs_reference"]["v22"]["evidence"] == "inconclusive"
    assert task["comparisons_vs_reference"]["v2"]["evidence"] == "inconclusive"
    assert task["comparisons_vs_reference"]["v2"]["regressed"] == 1


def test_exact_paired_uplift_and_missing_or_changed_stimuli(tmp_path):
    variants = {name: tmp_path / name for name in ("null", "positive")}
    _rows(variants["null"], [False] * 8)
    _rows(variants["positive"], [True] * 8)
    seed_map = {"schema_version": 1, "tasks": {"bell": list(range(8))}}
    task = compare_variants(variants, reference="null", seed_map=seed_map)["tasks"]["bell"]
    row = task["comparisons_vs_reference"]["positive"]
    assert row["two_sided_sign_test_p"] == 2 / 256
    assert row["evidence"] == "uplift"
    path = variants["positive"] / "0/manifest.json"
    data = json.loads(path.read_text())
    data["task"] = "wrong instruction"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Instruction mismatch"):
        compare_variants(variants, reference="null", seed_map=seed_map)
    data["task"] = "instruction-0"
    data["metadata"]["counterfactual_policy_decision"] = 0
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Protocol mismatch"):
        compare_variants(variants, reference="null", seed_map=seed_map)
    path.unlink()
    with pytest.raises(ValueError, match="seed set mismatch"):
        compare_variants(variants, reference="null", seed_map=seed_map)
