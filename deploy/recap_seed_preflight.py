"""Outcome-blind RoboTwin cohort construction; never runs the learned policy."""

from __future__ import annotations

import hashlib
import time

import numpy as np

from deploy.recap_instructions import INSTRUCTION_PROTOCOL, generate_task_instruction


def initial_observation_hashes(observation):
    arrays = {
        "state": observation["joint_action"]["vector"],
        **{name: observation["observation"][name]["rgb"]
           for name in ("head_camera", "left_camera", "right_camera")},
    }
    result = {}
    for name, value in arrays.items():
        value = np.ascontiguousarray(value)
        result[name] = hashlib.sha256(
            str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()
        ).hexdigest()
    return result


def preflight_environment_seeds(
    env, args, *, task, start_seed, count, generator, instruction_type="seen",
    max_candidates=None, setup_retries=2, progress=None,
):
    """Select first N expert-feasible initializations, independently of VLA outcomes.

    Both expert feasibility and a second rollout-style setup must pass. All
    rejected seeds/attempts are retained. Failure to fill the fixed budget is an
    infrastructure failure, not permission to shorten an evaluation cohort.
    """
    max_candidates = count * 20 if max_candidates is None else max_candidates
    if count <= 0 or max_candidates < count or setup_retries < 0:
        raise ValueError("Invalid cohort size/candidate budget/retry count")
    args = {**args, "eval_mode": True, "eval_video_log": False, "render_freq": 0}
    accepted, attempts = [], []
    started = time.time()
    for seed in range(start_seed, start_seed + max_candidates):
        for attempt in range(setup_retries + 1):
            record = {"seed": seed, "attempt": attempt + 1}
            try:
                try:
                    env.setup_demo(now_ep_num=len(accepted), seed=seed, is_test=True, **args)
                    info = env.play_once()
                    feasible = bool(env.plan_success and env.check_success())
                finally:
                    env.close_env()
                if not feasible:
                    raise RuntimeError("expert_feasibility_failed")
                try:
                    env.setup_demo(now_ep_num=len(accepted), seed=seed, is_test=True, **args)
                    text = generate_task_instruction(
                        task, seed, [info["info"]], instruction_type, count,
                        deterministic=True, generator=generator,
                    )
                    hashes = initial_observation_hashes(env.get_obs())
                finally:
                    env.close_env()
                accepted.append({"seed": seed, "instruction": text,
                                 "initial_observation_sha256": hashes})
                record["accepted"] = True
            except Exception as error:
                record.update(accepted=False, error_type=type(error).__name__, error=str(error)[:1000])
            attempts.append(record)
            if progress:
                progress(record)
            if record["accepted"]:
                break
        if len(accepted) == count:
            return {
                "schema_version": 1, "purpose": "expert_feasibility_only_no_learned_policy",
                "task": task, "start_seed": start_seed, "episodes": count,
                "max_candidates": max_candidates, "setup_retries": setup_retries,
                "instruction_protocol": INSTRUCTION_PROTOCOL, "instruction_type": instruction_type,
                "policy_rollouts": 0, "accepted": accepted, "attempts": attempts,
                "elapsed_seconds": time.time() - started,
            }
    raise RuntimeError(f"Preflight {task}: only {len(accepted)}/{count} feasible seeds in {max_candidates} candidates")
