import numpy as np
import pytest

from deploy.recap_seed_preflight import preflight_environment_seeds


class FakeEnvironment:
    def __init__(self, bad_seeds=()):
        self.bad_seeds = set(bad_seeds)
        self.calls = []
        self.plan_success = True

    def setup_demo(self, *, seed, now_ep_num, **kwargs):
        self.calls.append(("setup", seed, now_ep_num))
        self.seed = seed
        if seed in self.bad_seeds:
            raise RuntimeError("unstable")

    def play_once(self):
        self.calls.append(("expert", self.seed))
        return {"info": {}}

    def check_success(self):
        return True

    def close_env(self):
        self.calls.append(("close",))

    def get_obs(self):
        return {"joint_action": {"vector": np.zeros(14, dtype=np.float32)},
                "observation": {name: {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}
                                for name in ("head_camera", "left_camera", "right_camera")}}


def test_cohort_uses_only_prepolicy_feasibility_and_keeps_rejections():
    env = FakeEnvironment(bad_seeds=[100])
    report = preflight_environment_seeds(
        env, {}, task="bell", start_seed=100, count=2, setup_retries=1,
        generator=lambda *args: [{"seen": ["instruction"]}],
    )
    assert report["policy_rollouts"] == 0
    assert "success" not in report
    assert [row["seed"] for row in report["accepted"]] == [101, 102]
    assert [row["accepted"] for row in report["attempts"]] == [False, False, True, True]
    assert [call for call in env.calls if call[0] == "expert"] == [("expert", 101), ("expert", 102)]
    assert len(report["accepted"][0]["initial_observation_sha256"]) == 4
    assert env.calls.count(("close",)) == 6


def test_preflight_never_shrinks_a_failed_cohort():
    env = FakeEnvironment(bad_seeds=[100, 101])
    with pytest.raises(RuntimeError, match="only 0/2"):
        preflight_environment_seeds(
            env, {}, task="bell", start_seed=100, count=2, max_candidates=2,
            setup_retries=0, generator=lambda *args: [{"seen": ["instruction"]}],
        )
