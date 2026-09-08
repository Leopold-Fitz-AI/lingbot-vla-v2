import ast
import hashlib
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from deploy.recap_evaluator_context import capture_evaluator_context, restore_evaluator_context
from deploy.recap_locked_cohort import LOCKED_COHORT_PROTOCOL, CohortProtocolError, load_locked_cohort
from deploy.recap_seed_preflight import preflight_environment_seeds
from tests.test_recap_seed_preflight import FakeEnvironment


def encode(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def cohort_files(tmp_path):
    report = preflight_environment_seeds(
        FakeEnvironment(),
        {},
        task="bell",
        start_seed=100,
        count=2,
        generator=lambda *args: [{"seen": ["literal instruction"]}],
    )
    preflight, lock, manifest = (tmp_path / name for name in ("preflight.json", "lock.json", "cohorts.json"))
    encode(preflight, report)
    digest = hashlib.sha256(preflight.read_bytes()).hexdigest()
    encode(lock, {"sha256": digest})
    encode(
        manifest,
        {
            "schema_version": 1,
            "task_config": "demo_clean",
            "tasks": {"bell": {"path": str(preflight), "sha256": digest}},
        },
    )
    kwargs = dict(
        task="bell",
        task_config="demo_clean",
        seeds=[100, 101],
        instructions=[r["instruction"] for r in report["accepted"]],
        count=2,
    )
    return manifest, preflight, lock, report, kwargs


def test_verified_cohort_and_initial_observation(cohort_files):
    manifest, _, _, report, kwargs = cohort_files
    cohort = load_locked_cohort(manifest, **kwargs)
    metadata = cohort.initial_metadata(100, FakeEnvironment().get_obs())
    assert metadata["eligibility_protocol"] == LOCKED_COHORT_PROTOCOL
    assert metadata["initial_observation_sha256"] == report["accepted"][0]["initial_observation_sha256"]
    observation = FakeEnvironment().get_obs()
    observation["joint_action"]["vector"][0] = 1e-5
    with pytest.raises(CohortProtocolError, match="initial observation mismatch"):
        cohort.initial_metadata(100, observation)


@pytest.mark.parametrize("change", ["task", "config", "seeds", "order", "shorten", "instructions", "missing_map"])
def test_no_unverified_or_substituted_cohort(cohort_files, change):
    manifest, _, _, _, kwargs = cohort_files
    if change == "task":
        kwargs["task"] = "different"
    if change == "config":
        kwargs["task_config"] = "different"
    if change == "seeds":
        kwargs["seeds"] = [100, 102]
    if change == "order":
        kwargs["seeds"] = [101, 100]
    if change == "shorten":
        kwargs.update(seeds=[100], count=1)
    if change == "instructions":
        kwargs["instructions"] = ["different", "different"]
    if change == "missing_map":
        kwargs["seeds"] = None
    with pytest.raises(CohortProtocolError):
        load_locked_cohort(manifest, **kwargs)


@pytest.mark.parametrize("which", ["report", "lock", "manifest"])
def test_tampering_is_fatal(cohort_files, which):
    manifest, preflight, lock, _, kwargs = cohort_files
    if which == "report":
        preflight.write_text(preflight.read_text() + " ")
    if which == "lock":
        encode(lock, {"sha256": "0" * 64})
    if which == "manifest":
        payload = json.loads(manifest.read_text())
        payload["tasks"]["bell"]["sha256"] = "0" * 64
        encode(manifest, payload)
    with pytest.raises(CohortProtocolError, match="checksum"):
        load_locked_cohort(manifest, **kwargs)


def test_no_expert_evidence_cannot_be_forged_by_changing_only_seed_list(cohort_files):
    manifest, preflight, lock, report, kwargs = cohort_files
    report["attempts"] = []
    encode(preflight, report)
    digest = hashlib.sha256(preflight.read_bytes()).hexdigest()
    encode(lock, {"sha256": digest})
    mapping = json.loads(manifest.read_text())
    mapping["tasks"]["bell"]["sha256"] = digest
    encode(manifest, mapping)
    with pytest.raises(CohortProtocolError, match="accepted expert attempt"):
        load_locked_cohort(manifest, **kwargs)


class RolloutEnvironment(FakeEnvironment):
    step_lim = 1
    eval_video_path = None
    render_freq = 0

    def setup_demo(self, **kwargs):
        super().setup_demo(**kwargs)
        self.take_action_cnt = 0
        self.eval_success = False

    def close_env(self, **kwargs):
        super().close_env()

    def set_instruction(self, instruction):
        self.instruction = instruction

    def get_instruction(self):
        return self.instruction

    def take_action(self, action):
        self.take_action_cnt += 1


class Policy:
    def __init__(self):
        self.calls = []

    def infer(self, data):
        self.calls.append(data)
        return {} if data.get("reset") else {"action": np.zeros((1, 14)), "server_timing": {}}


def client_function(monkeypatch, kwargs):
    path = Path(__file__).parents[1] / "experiment/robotwin/eval_policy_client_lingbotvla.py"
    fn = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "eval_policy"
    )
    namespace = dict(
        os=os,
        np=np,
        Path=Path,
        CohortProtocolError=CohortProtocolError,
        capture_evaluator_context=capture_evaluator_context,
        restore_evaluator_context=restore_evaluator_context,
        mapped_environment_seeds=lambda task: kwargs["seeds"],
        mapped_task_instruction=lambda task, seed: kwargs["instructions"][kwargs["seeds"].index(seed)],
        mapped_counterfactual_decision=lambda task: 0,
        deterministic_instructions_enabled=lambda: False,
        UnStableError=type("UnStableError", (Exception,), {}),
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    monkeypatch.setenv("RECAP_TASK_INSTRUCTION_MAP", "mock-literal-map")
    monkeypatch.delenv("RECAP_TASK_INSTRUCTION", raising=False)
    return namespace[fn.name]


@pytest.mark.parametrize("locked", [True, False])
def test_actual_client_never_rescreens_locked_seeds_and_keeps_policy_failures(cohort_files, monkeypatch, locked):
    manifest, _, _, _, kwargs = cohort_files
    if locked:
        monkeypatch.setenv("RECAP_COHORT_MANIFEST", str(manifest))
    else:
        monkeypatch.delenv("RECAP_COHORT_MANIFEST", raising=False)
    records = []

    class Recorder:
        def __init__(self, *args, **kw):
            records.append(kw)

        def record_step(self, *args, **kw):
            pass

        def finalize(self, **kw):
            assert not kw["success"]

    module = ModuleType("script.deploy.recap_rollout_recorder")
    module.RecapEpisodeRecorder = Recorder
    monkeypatch.setitem(__import__("sys").modules, module.__name__, module)
    env, model = RolloutEnvironment(), Policy()
    fn = client_function(monkeypatch, kwargs)
    _, successes, rows = fn(
        "bell",
        env,
        dict(
            task_name="bell",
            policy_name="ACT",
            task_config="demo_clean",
            render_freq=0,
            clear_cache_freq=5,
            ckpt_setting=None,
        ),
        model,
        100,
        test_num=2,
        usr_args={"robo_name": "robotwin", "recap_rollout_dir": "unused"},
    )
    assert successes == 0 and [r["seed"] for r in rows] == [100, 101]
    assert len(model.calls) == 4  # reset + action for EACH state, even after failures
    assert len([call for call in env.calls if call[0] == "expert"]) == (0 if locked else 2)
    assert all(r["metadata"]["expert_rollouts_before_policy"] == (0 if locked else 1) for r in records)
    if locked:
        assert all(r["metadata"]["eligibility_protocol"] == LOCKED_COHORT_PROTOCOL for r in records)


def test_wrong_initial_state_stops_before_any_policy_call(cohort_files, monkeypatch):
    manifest, _, _, _, kwargs = cohort_files
    monkeypatch.setenv("RECAP_COHORT_MANIFEST", str(manifest))
    fn = client_function(monkeypatch, kwargs)
    env, model = RolloutEnvironment(), Policy()
    observation = env.get_obs()
    observation["observation"]["head_camera"]["rgb"][0, 0, 0] = 1
    env.get_obs = lambda: observation
    with pytest.raises(CohortProtocolError, match="initial observation mismatch"):
        fn(
            "bell",
            env,
            dict(task_name="bell", policy_name="ACT", task_config="demo_clean", render_freq=0, clear_cache_freq=5),
            model,
            100,
            test_num=2,
            usr_args={"robo_name": "robotwin"},
        )
    assert model.calls == []
