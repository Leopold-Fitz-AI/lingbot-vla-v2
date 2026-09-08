import random

import numpy as np
import pytest

from deploy.recap_instructions import DETERMINISTIC_CANDIDATES, generate_task_instruction


def _generator(task, episodes, count):
    # Both candidate length and the unseen candidates' RNG offset depend on count,
    # as in RoboTwin. Also exercise NumPy RNG isolation and input mutation safety.
    episodes[0]["mutated"] = True
    return [{kind: [f"{task}-{random.random()}-{np.random.rand()}" for _ in range(count)]
             for kind in ("seen", "unseen")}]


@pytest.mark.parametrize("kind", ["seen", "unseen"])
def test_deterministic_text_does_not_depend_on_episode_count_or_rng_history(kind):
    info = [{"arm": "left"}]
    texts = []
    for count in (3, 5, 12, 20, 30, 100):
        random.seed(count)
        np.random.seed(count)
        python_before, numpy_before = random.getstate(), np.random.get_state()
        texts.append(generate_task_instruction(
            "click_bell", 3300002, info, kind, count,
            deterministic=True, generator=_generator,
        ))
        assert random.getstate() == python_before
        numpy_after = np.random.get_state()
        assert numpy_before[0] == numpy_after[0]
        assert np.array_equal(numpy_before[1], numpy_after[1])
        assert numpy_before[2:] == numpy_after[2:]
    assert len(set(texts)) == 1
    assert info == [{"arm": "left"}]


def test_deterministic_generation_uses_fixed_budget_and_distinct_environment_seeds():
    budgets = []

    def generator(task, episodes, count):
        budgets.append(count)
        return _generator(task, episodes, count)

    values = [generate_task_instruction(
        "click_bell", seed, [{}], "seen", 3, deterministic=True, generator=generator,
    ) for seed in (3300002, 3400006)]
    assert budgets == [DETERMINISTIC_CANDIDATES] * 2
    assert values[0] != values[1]


def test_legacy_random_mode_keeps_requested_budget():
    budgets = []

    def generator(task, episodes, count):
        budgets.append(count)
        return [{"seen": ["legacy"]}]

    assert generate_task_instruction(
        "bell", 42, [{}], "seen", 20, deterministic=False, generator=generator,
    ) == "legacy"
    assert budgets == [20]


def test_generator_failure_restores_rng_states():
    random.seed(10)
    np.random.seed(10)
    python_before, numpy_before = random.getstate(), np.random.get_state()

    def generator(*args):
        random.random()
        np.random.rand()
        raise RuntimeError("bad description")

    with pytest.raises(RuntimeError, match="bad description"):
        generate_task_instruction(
            "bell", 42, [{}], "seen", 3, deterministic=True, generator=generator,
        )
    assert random.getstate() == python_before
    assert np.array_equal(np.random.get_state()[1], numpy_before[1])


def test_empty_candidates_fail_explicitly():
    with pytest.raises(ValueError, match="No instruction candidates"):
        generate_task_instruction(
            "bell", 42, [{}], "seen", 3,
            deterministic=True, generator=lambda *args: [{"seen": []}],
        )
