import copy

import pytest

from scripts.recap_check_repaired_policy import COUNT, TASKS, resources, summarize


def counts(delta, improved=0, regressed=0):
    return {
        task: {
            "episodes": COUNT,
            "positive_success": 20,
            "null_success": 20,
            "improved": improved,
            "regressed": regressed,
            "ties": COUNT - improved - regressed,
            "delta": delta,
            "two_sided_p": 1.0,
        }
        for task in TASKS
    }


def test_summary_fixed_priority_gate_and_reproducible_bootstrap():
    pairs = {task: [1] * 2 + [0] * (COUNT - 2) for task in TASKS}
    first = summarize(pairs, counts(0.05, improved=2))
    second = summarize(pairs, counts(0.05, improved=2))
    assert first == second
    assert first["macro_delta_pp"] == pytest.approx(5.0)
    assert first["worth_further_investigation"] is True
    assert first["policy_episodes"] == 240


def test_summary_rejects_incomplete_and_any_net_regression():
    pairs = {task: [1] * 2 + [0] * (COUNT - 2) for task in TASKS}
    pairs[TASKS[0]] = [-1] + [0] * (COUNT - 1)
    report = summarize(pairs, counts(0.0))
    assert report["worth_further_investigation"] is False
    broken = copy.deepcopy(pairs)
    broken[TASKS[1]].pop()
    with pytest.raises(ValueError, match="Complete"):
        summarize(broken, counts(0.0))
    broken = copy.deepcopy(pairs)
    broken[TASKS[1]][0] = 2
    with pytest.raises(ValueError, match="Invalid"):
        summarize(broken, counts(0.0))


def test_resource_gate_only_accepts_idle_authorized_gpus(monkeypatch):
    calls = []

    def output(command, text):
        calls.append(command)
        if "memory.used" in command[1]:
            return "0, 20000\n1, 20000\n2, 20000\n3, 20000\n4, 90000\n5, 1\n6, 2\n7, 3\n"
        if "compute-apps" in command[1]:
            return "GPU-0, 123\n"
        return "0, GPU-0\n1, GPU-1\n2, GPU-2\n3, GPU-3\n4, GPU-4\n5, GPU-5\n6, GPU-6\n7, GPU-7\n"

    monkeypatch.setattr("scripts.recap_check_repaired_policy.subprocess.check_output", output)
    assert resources()["memory_mib"][7] == 3
    assert len(calls) == 3

    def busy(command, text):
        if "memory.used" in command[1]:
            return "0, 1\n1, 1\n2, 1\n3, 1\n4, 1\n5, 1025\n6, 1\n7, 1\n"
        return ""

    monkeypatch.setattr("scripts.recap_check_repaired_policy.subprocess.check_output", busy)
    with pytest.raises(RuntimeError, match="not idle"):
        resources()
