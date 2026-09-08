import re
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("common_noise,expected", [("true", "900\n900\n900\n"),
                                                   ("false", "900\n901\n903\n")])
def test_server_and_recorded_policy_seed_share_slot_rule(common_noise, expected):
    source = (Path(__file__).parents[1] / "experiment/robotwin/start_robotwin_infer_and_eval.sh").read_text()
    helper = re.search(r"^policy_seed_for_slot\(\) \{\n.*?^\}", source, re.M | re.S)
    assert helper is not None
    assert source.count('slot_policy_seed=$(policy_seed_for_slot "$slot")') == 2
    result = subprocess.run(
        ["bash", "-c", helper.group() + f"\npolicy_seed=900; common_noise_per_episode={common_noise}; "
         "for slot in 0 1 3; do policy_seed_for_slot $slot; done"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout == expected


@pytest.mark.parametrize("exit_code,expected", [(1, "retry"), (78, "stop")])
def test_locked_state_mismatches_are_not_retried(exit_code, expected):
    source = (Path(__file__).parents[1] / "experiment/robotwin/start_robotwin_infer_and_eval.sh").read_text()
    condition = next(line.strip() for line in source.splitlines()
                     if 'task_retries[$task_name]} -lt $max_retries' in line)
    result = subprocess.run(
        ["bash", "-c", 'declare -A task_retries=([bell]=1); task_name=bell; max_retries=3; '
         + f'exit_code={exit_code}; ' + condition + ' echo retry; else echo stop; fi'],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == expected
