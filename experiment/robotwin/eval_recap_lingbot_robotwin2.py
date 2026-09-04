"""Run RoboTwin 2.0 against LingBot-VLA's lightweight WebSocket server.

RoboTwin's current XPolicyLab evaluator speaks a framed WebSocket protocol,
while LingBot-VLA serves the simpler OpenPI-style msgpack protocol.  This
entry point adapts the two without requiring XPolicyLab on the inference
host.  It also records exact generated and executed action prefixes as RECAP
rollout bundles.

This file is designed to be copied into ``<RoboTwin>/scripts`` together with
``deploy/recap_rollout_recorder.py`` as ``scripts/recap_rollout_recorder.py``.
It reuses the simulator and evaluation helpers from
``scripts/eval_policy_xpolicylab.py``.
"""

from __future__ import annotations

import functools
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import websockets.exceptions
import websockets.sync.client

try:
    from scripts import eval_policy_xpolicylab as robotwin_eval
except ImportError:
    import eval_policy_xpolicylab as robotwin_eval

try:
    from scripts.recap_rollout_recorder import RecapEpisodeRecorder
except ImportError:
    from recap_rollout_recorder import RecapEpisodeRecorder


def _pack_array(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def _unpack_array(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    ndarray_key = b"__ndarray__" if b"__ndarray__" in value else "__ndarray__"
    generic_key = b"__npgeneric__" if b"__npgeneric__" in value else "__npgeneric__"
    if ndarray_key in value:
        key = (lambda name: name.encode() if ndarray_key == b"__ndarray__" else name)
        return np.ndarray(
            buffer=value[key("data")],
            dtype=np.dtype(value[key("dtype")]),
            shape=value[key("shape")],
        )
    if generic_key in value:
        key = (lambda name: name.encode() if generic_key == b"__npgeneric__" else name)
        return np.dtype(value[key("dtype")]).type(value[key("data")])
    return value


_Packer = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class LingbotWsModelClient:
    """Expose the XPolicyLab ``call`` API over LingBot's msgpack protocol."""

    # RoboTwin uses this marker to send prepare_case/trial_end callbacks.
    _robotwin_protocol = "ws"

    def __init__(
        self,
        *,
        host: str,
        port: int,
        rollout_dir: str | os.PathLike[str],
        run_id: str,
        checkpoint: str,
        recap_condition: str = "positive",
        recap_cfg_scale: float = 1.0,
        robo_name: str = "robotwin",
        connect_timeout_s: float = 1200.0,
    ) -> None:
        self._uri = f"ws://{host}:{int(port)}"
        self._packer = _Packer()
        self._ws = None
        self._closed = False
        self._ws = self._connect(connect_timeout_s)
        metadata_frame = self._ws.recv()
        if isinstance(metadata_frame, str):
            raise RuntimeError(f"LingBot server returned text metadata: {metadata_frame}")
        self.server_metadata = _unpackb(metadata_frame)
        if not isinstance(self.server_metadata, dict):
            raise TypeError("LingBot server metadata frame must decode to a dictionary")
        self.rollout_dir = Path(rollout_dir).expanduser().resolve()
        self.rollout_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.checkpoint = str(checkpoint)
        self.recap_condition = str(recap_condition)
        self.recap_cfg_scale = float(recap_cfg_scale)
        self.robo_name = str(robo_name)
        self._latest_observation: dict[str, Any] | None = None
        self._recorder: RecapEpisodeRecorder | None = None
        self._pending: dict[str, Any] | None = None
        self._executed_actions: list[np.ndarray] = []
        self._environment_steps = 0

    def _connect(self, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    ping_interval=None,
                    ping_timeout=None,
                    open_timeout=10,
                )
            except (ConnectionRefusedError, OSError, TimeoutError, websockets.exceptions.WebSocketException) as error:
                last_error = error
                time.sleep(2)
        raise TimeoutError(f"Timed out waiting for LingBot server at {self._uri}: {last_error}")

    def _infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        self._ws.send(self._packer.pack(dict(observation)))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"LingBot inference server error:\n{response}")
        result = _unpackb(response)
        if not isinstance(result, dict):
            raise TypeError(f"LingBot server returned {type(result)!r}, expected dict")
        return result

    @staticmethod
    def _format_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
        vision = observation.get("vision", {})
        state = observation.get("state", {})

        def color(name: str) -> np.ndarray:
            camera = vision.get(name)
            if not isinstance(camera, Mapping) or "color" not in camera:
                raise KeyError(f"Missing RoboTwin camera color: {name}")
            image = np.asarray(camera["color"])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Camera {name} must be HWC RGB, got {image.shape}")
            return image

        def vector(name: str) -> np.ndarray:
            if name not in state:
                raise KeyError(f"Missing RoboTwin state field: {name}")
            return np.asarray(state[name], dtype=np.float32).reshape(-1)

        joint_state = np.concatenate(
            [
                vector("left_arm_joint_state"),
                vector("left_ee_joint_state"),
                vector("right_arm_joint_state"),
                vector("right_ee_joint_state"),
            ]
        ).astype(np.float32, copy=False)
        if joint_state.shape != (14,):
            raise ValueError(f"LingBot RoboTwin state must have shape (14,), got {joint_state.shape}")

        return {
            "observation.images.cam_high": color("cam_head"),
            "observation.images.cam_left_wrist": color("cam_left_wrist"),
            "observation.images.cam_right_wrist": color("cam_right_wrist"),
            "observation.state": joint_state,
            "task": str(observation.get("instruction", "")),
        }

    def _prepare_case(self, metadata: Mapping[str, Any]) -> None:
        if self._recorder is not None:
            raise RuntimeError("Previous RECAP episode was not finalized")
        task_name = str(metadata.get("task_name", "unknown-task"))
        safe_task_name = re.sub(r"[^A-Za-z0-9._-]+", "-", task_name).strip("-.") or "unknown-task"
        action_type = str(metadata.get("action_type", "joint")).lower()
        if action_type not in {"joint", "qpos"}:
            raise ValueError(
                "LingBot RoboTwin rollout adapter supports only 14-D joint/qpos actions; "
                f"got action_type={action_type!r}"
            )
        seed = int(metadata.get("seed", 0))
        instruction = str(metadata.get("instruction", task_name))
        episode_id = f"{safe_task_name}-{self.run_id}-seed-{seed}"
        self._recorder = RecapEpisodeRecorder(
            self.rollout_dir / safe_task_name,
            episode_id=episode_id,
            task=instruction,
            seed=seed,
            metadata={
                "task_name": task_name,
                "checkpoint": self.checkpoint,
                "recap_condition": self.recap_condition,
                "recap_cfg_scale": self.recap_cfg_scale,
                "robo_name": self.robo_name,
                "policy_protocol": "lingbot_msgpack_websocket",
                "action_type": action_type,
                "state_dim": 14,
                "action_dim": 14,
                "required_cameras": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
            },
        )
        self._latest_observation = None
        self._pending = None
        self._executed_actions = []
        self._environment_steps = 0

    def note_executed_action(self, action: Any) -> None:
        """Record one 14-D joint action after RoboTwin advances its step counter."""
        if self._pending is None:
            return
        executed = np.asarray(action, dtype=np.float32).reshape(-1)
        generated = np.asarray(self._pending["generated_action"])
        if executed.shape != (14,) or generated.shape[-1] != 14:
            raise ValueError(
                "Generated and executed RoboTwin actions must both use the 14-D joint space; "
                f"got generated={generated.shape}, executed={executed.shape}"
            )
        self._executed_actions.append(executed.copy())

    def _flush_pending(self, *, terminated: bool, truncated: bool) -> None:
        if self._pending is None:
            return
        if self._recorder is None:
            raise RuntimeError("Cannot record a policy decision without an active episode")
        generated = np.asarray(self._pending["generated_action"], dtype=np.float32)
        if generated.ndim == 1:
            generated = generated[None, :]
        if generated.ndim != 2 or generated.shape[-1] != 14:
            raise ValueError(f"Generated RoboTwin action chunk must have shape [T,14], got {generated.shape}")
        if self._executed_actions:
            executed = np.stack(self._executed_actions, axis=0)
        else:
            executed = np.empty((0, generated.shape[-1]), dtype=generated.dtype)
        if executed.shape[0] > generated.shape[0]:
            raise RuntimeError(
                f"Executed prefix length {executed.shape[0]} exceeds generated chunk {generated.shape[0]}"
            )
        self._recorder.record_step(
            self._pending["observation"],
            generated,
            executed,
            env_step_before=self._environment_steps,
            env_step_after=self._environment_steps + executed.shape[0],
            terminated=terminated,
            truncated=truncated,
            info={"server_timing": self._pending.get("server_timing", {})},
        )
        self._environment_steps += executed.shape[0]
        self._pending = None
        self._executed_actions = []

    def _abort_episode(self, reason: str, *, recording_error: str | None = None) -> None:
        """Atomically finalize an incomplete episode instead of leaving orphan NPZs."""
        if self._recorder is None:
            return
        pending_generated = 0
        pending_executed = len(self._executed_actions)
        if self._pending is not None:
            generated = np.asarray(self._pending.get("generated_action"))
            pending_generated = int(generated.shape[0]) if generated.ndim >= 2 else 1
        recorder = self._recorder
        self._recorder = None
        self._pending = None
        self._executed_actions = []
        metadata = {
            "environment_steps": self._environment_steps,
            "discarded_pending_generated_actions": pending_generated,
            "discarded_pending_executed_actions": pending_executed,
        }
        if recording_error:
            metadata["recording_error"] = recording_error
        recorder.finalize(success=False, terminal_reason=reason, metadata=metadata)

    def call(self, *, func_name: str, obs: Mapping[str, Any] | None = None) -> Any:
        if func_name == "prepare_case":
            self._prepare_case(dict(obs or {}))
            return None
        if func_name == "reset":
            return self._infer({"reset": True, "robo_name": self.robo_name})
        if func_name == "update_obs":
            if obs is None:
                raise ValueError("update_obs requires obs")
            self._latest_observation = self._format_observation(obs)
            return None
        if func_name == "get_action":
            if self._latest_observation is None:
                raise RuntimeError("get_action called before update_obs")
            # A new policy decision means the previous generated chunk has
            # finished executing without terminating the episode.
            self._flush_pending(terminated=False, truncated=False)
            response = self._infer(self._latest_observation)
            if "action" not in response:
                raise KeyError(f"LingBot response has no 'action' key: {response.keys()}")
            action = np.asarray(response["action"], dtype=np.float32)
            if action.ndim not in (1, 2) or action.shape[-1] != 14:
                raise ValueError(f"LingBot joint action must be [14] or [T,14], got {action.shape}")
            self._pending = {
                "observation": self._latest_observation,
                "generated_action": action.copy(),
                "server_timing": response.get("server_timing", {}),
            }
            self._executed_actions = []
            return action
        if func_name == "trial_end":
            metadata = dict(obs or {})
            success = bool(metadata.get("success", False))
            self._flush_pending(terminated=success, truncated=not success)
            if self._recorder is not None:
                self._recorder.finalize(
                    success=success,
                    terminal_reason="success" if success else "step_limit_or_rollout_error",
                    metadata={"environment_steps": self._environment_steps},
                )
                self._recorder = None
            return None
        raise NotImplementedError(f"Unsupported policy call: {func_name}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        recording_error: Exception | None = None
        if self._recorder is not None:
            try:
                self._flush_pending(terminated=False, truncated=True)
                recorder = self._recorder
                self._recorder = None
                recorder.finalize(
                    success=False,
                    terminal_reason="client_closed_before_trial_end",
                    metadata={"environment_steps": self._environment_steps},
                )
            except Exception as error:
                recording_error = error
                self._abort_episode("rollout_recording_error", recording_error=repr(error))
        if self._ws is not None:
            self._ws.close()
            self._ws = None
        if recording_error is not None:
            raise RuntimeError("RECAP episode recording failed during client close") from recording_error


def _build_policy_client(usr_args: dict[str, Any]) -> LingbotWsModelClient:
    rollout_dir = usr_args.get("recap_rollout_dir")
    if not rollout_dir:
        raise ValueError("additional_info must define recap_rollout_dir=<path>")
    return LingbotWsModelClient(
        host=str(usr_args.get("host", "127.0.0.1")),
        port=int(usr_args["port"]),
        rollout_dir=str(rollout_dir),
        run_id=str(usr_args.get("recap_run_id", "run")),
        checkpoint=str(usr_args.get("ckpt_setting") or usr_args.get("ckpt_name") or "unknown"),
        recap_condition=str(usr_args.get("recap_condition", "positive")),
        recap_cfg_scale=float(usr_args.get("recap_cfg_scale", 1.0)),
        robo_name=str(usr_args.get("robo_name", "robotwin")),
    )


_ORIGINAL_EVAL_REMOTE_POLICY = robotwin_eval.eval_remote_policy


def _eval_remote_policy_with_action_tracking(
    task_name,
    task_env,
    args,
    model_client,
    st_seed,
    **kwargs,
):
    """Notify the recorder only after each simulator action succeeds."""
    original_take_action = task_env.take_action

    def tracked_take_action(action, *action_args, **action_kwargs):
        before = getattr(task_env, "take_action_cnt", None)
        result = original_take_action(action, *action_args, **action_kwargs)
        after = getattr(task_env, "take_action_cnt", None)
        if isinstance(before, (int, np.integer)) and isinstance(after, (int, np.integer)):
            if int(after) == int(before) + 1:
                model_client.note_executed_action(action)
            elif int(after) != int(before):
                raise RuntimeError(
                    f"RoboTwin action counter advanced by {int(after) - int(before)}, expected exactly 1"
                )
        else:
            # Older RoboTwin versions expose no counter; successful return is
            # their only acceptance signal.
            model_client.note_executed_action(action)
        return result

    task_env.take_action = tracked_take_action
    try:
        return _ORIGINAL_EVAL_REMOTE_POLICY(
            task_name,
            task_env,
            args,
            model_client,
            st_seed,
            **kwargs,
        )
    finally:
        task_env.take_action = original_take_action


def _prepare_policy_case_strict(model_client, task_name, seed, instruction, action_type) -> None:
    model_client.call(
        func_name="prepare_case",
        obs={
            "task_name": task_name,
            "seed": int(seed),
            "instruction": instruction,
            "action_type": action_type,
        },
    )


def _notify_trial_end_strict(model_client, task_name, seed, success) -> None:
    model_client.call(
        func_name="trial_end",
        obs={"task_name": task_name, "seed": int(seed), "success": bool(success)},
    )


def main() -> None:
    robotwin_eval.build_policy_client = _build_policy_client
    robotwin_eval.eval_remote_policy = _eval_remote_policy_with_action_tracking
    robotwin_eval.prepare_policy_case = _prepare_policy_case_strict
    robotwin_eval.notify_trial_end = _notify_trial_end_strict
    instruction_override = os.environ.get("RECAP_TASK_INSTRUCTION", "").strip()
    if instruction_override:
        robotwin_eval.build_instruction = (
            lambda args, episode_info, instruction_type, test_num: instruction_override
        )

    # Match RoboTwin's stock entry point: verify Vulkan rendering before eval.
    from test_render import Sapien_TEST

    Sapien_TEST()
    parsed_args = robotwin_eval.parse_args()
    if robotwin_eval.parse_bool(parsed_args.get("eval_batch", False)):
        raise ValueError("RECAP rollout recording currently requires eval_batch=false")
    robotwin_eval.main(parsed_args)


if __name__ == "__main__":
    main()
