#!/usr/bin/env python3
"""Outcome-blind, bounded resource wait followed by ONE frozen feature probe.

Never starts/retries the study, never reads final outcomes, never runs training.
Waiting for hardware is not a retry of a model run: the launch latch is exclusive
and permanent, and every child exit (including failure) terminates this observer.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from lingbotvla.recap.evaluation import write_json
from scripts.recap_audit_backend_parity import require_idle_gpu


class ProbeWaitFailure(RuntimeError):
    pass


def readiness(study, gpu):
    status = json.loads((study / "status.json").read_text())
    stage = status.get("stage")
    if stage == "stopped" or str(status.get("state", "")).startswith("infrastructure_or_protocol_failure"):
        raise ProbeWaitFailure("Frozen study failed; manual audit required, no probe launched")
    if stage != "complete":
        return "waiting_for_frozen_study"
    try:
        require_idle_gpu(gpu, study)
    except RuntimeError as exc:
        if "occupied" not in str(exc):
            raise
        return "waiting_for_gpu"
    return "ready"


def wait_and_run(output, study, gpu, *, poll_seconds=60, max_wait_seconds=172800):
    if not math.isfinite(poll_seconds) or not poll_seconds > 0 or not 0 < max_wait_seconds <= 172800:
        raise ValueError("Wait interval must be finite/positive and deadline at most 48 hours")
    if gpu not in (4, 5, 6, 7) or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise ValueError("Only one explicitly authorized GPU 4–7 may be queued")
    if not (output / "seal.json").is_file():
        raise ValueError("Freeze the probe input lock before queuing")
    if any(
        (output / name).exists()
        for name in ("launch_once.json", "started.json", "queue_failure.json", "queue_result.json")
    ):
        raise ProbeWaitFailure("An earlier queue/run exists; no restart is permitted")
    deadline = time.monotonic() + max_wait_seconds
    while True:
        state = readiness(study, gpu)
        write_json(
            output / "queue_status.json",
            {"state": state, "gpu": gpu, "updated_unix": time.time(), "policy_episodes": 0, "optimizer_steps": 0},
        )
        print(json.dumps({"state": state, "gpu": gpu}), flush=True)
        if state == "ready":
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProbeWaitFailure("Resource wait expired; no model run launched")
        time.sleep(min(poll_seconds, remaining))
    code = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(code / "scripts/recap_audit_backend_parity.py"),
        "--run",
        "--gpu",
        str(gpu),
        "--active-study",
        str(study),
        "--output",
        str(output),
    ]
    # This is a launch latch, not a success marker. Preserve it even if the
    # process dies between creation and exec; manual diagnosis beats a duplicate.
    with (output / "launch_once.json").open("x") as handle:
        json.dump({"command": command, "launched_unix": time.time(), "gpu": gpu}, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    write_json(output / "queue_status.json", {"state": "probe_launched_once", "gpu": gpu})
    with (output / "probe.log").open("x") as log:
        process = subprocess.run(command, cwd=code, stdout=log, stderr=subprocess.STDOUT, check=False)
    result = {
        "state": "probe_process_exited",
        "returncode": process.returncode,
        "retries": 0,
        "training_launched": False,
        "controller_restarted": False,
    }
    write_json(output / "queue_result.json", result, immutable=True)
    write_json(output / "queue_status.json", result)
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active-study", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True, choices=(4, 5, 6, 7))
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--max-wait-seconds", type=float, default=172800)
    args = parser.parse_args()
    with (args.output / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            code = wait_and_run(
                args.output,
                args.active_study,
                args.gpu,
                poll_seconds=args.poll_seconds,
                max_wait_seconds=args.max_wait_seconds,
            )
        except Exception as exc:
            if not (args.output / "queue_failure.json").exists():
                write_json(args.output / "queue_failure.json", {"error": repr(exc)}, immutable=True)
            raise
    raise SystemExit(code)


if __name__ == "__main__":
    main()
