#!/usr/bin/env python3
"""Supervise the preregistered 50-task study; never tune on final outcomes.

Run with the inference conda Python and PYTHONPATH pointing to this checkout.
--initialize freezes a private code/adapter snapshot; subsequent invocations
resume the SAME study and SAME cohorts. Final evaluation cannot run before a
checksum-frozen registry has passed independent validation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

from deploy.recap_locked_cohort import LOCKED_COHORT_PROTOCOL, load_locked_cohort
from lingbotvla.recap.evaluation import (
    final_statistics,
    holm_adjust,
    initial_drift,
    paired_counts,
    power_sensitivity,
    recorded_initial_hashes,
    sha256,
    validate_runtime,
    write_json,
)


TARGETS = ["click_bell", "hanging_mug", "place_can_basket", "stack_bowls_three"]
PROTOCOL = {"policy_seed": 900, "continuation_policy_seed": 300, "counterfactual_policy_decision": 0,
            "common_noise_per_episode": True, "instruction_protocol": "task-seed-v2-fixed-candidates32",
            "task_config": "demo_clean", "use_bf16": False, "use_compile": False,
            "deterministic_algorithms": True, "moe_reduction": "stable_pack_fixed_routing_rank_sum",
            "eligibility_protocol": LOCKED_COHORT_PROTOCOL,
            "cfg_scale": 1.0, "chunk_size": 50}
SOURCE_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".json"}


def read_json(path):
    return json.loads(Path(path).read_text())


def code_hash(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in SOURCE_SUFFIXES and "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(root)).encode() + b"\0" + sha256(path).encode())
    return digest.hexdigest()


def inherit_cohorts(study, previous, plan):
    """Explicit amendment: copy complete consumed cohorts, never policy outcomes."""
    old = read_json(previous / "plan.json")
    if sha256(previous / "plan.json") != read_json(previous / "plan_lock.json")["sha256"]:
        raise ValueError("Predecessor plan changed")
    if read_json(previous / "status.json")["state"] != "infrastructure_or_protocol_failure":
        raise ValueError("Predecessor must be stopped on a recorded infrastructure failure")
    if ((previous / "final_opened.json").exists()
            or any(p.name not in {"s32", "s42"} for p in (previous / "cohorts").iterdir())
            or (previous / "evaluations/confirm").exists()):
        raise ValueError("Cannot reuse confirmation/final cohorts as a recovery diagnostic")
    for key in ("all_tasks", "target_tasks", "gpus", "task_gpu", "target_gpu", "base_files", "wrapper_sha256",
                "control", "screen", "confirm", "final", "confirmation_family", "confirmation_holm_alpha",
                "confirmation_min_delta", "bootstrap_replicates", "bootstrap_seed"):
        if old[key] != plan[key]:
            raise ValueError(f"Recovery changed the frozen study design: {key}")
    if old["protocol"].get("eligibility_protocol") is not None:
        raise ValueError("This amendment cannot restart an already locked-cohort protocol")
    if any(plan["protocol"].get(key) != value for key, value in old["protocol"].items()):
        raise ValueError("Recovery changed the policy numerical protocol")
    def candidates(data):
        return {task: [{k: v for k, v in entry.items() if k != "path"} for entry in entries]
                for task, entries in data["candidates"].items()}
    if candidates(old) != candidates(plan):
        raise ValueError("Recovery changed candidate weights/windows/priority")
    files = []
    for stage, index in (("control", 32), ("screen", 42)):
        for task in plan["target_tasks"]:
            source = previous / "cohorts" / f"s{index}" / task
            expected = read_json(source / "lock.json")["sha256"]
            report = read_json(source / "preflight.json")
            if (sha256(source / "preflight.json") != expected or report["task"] != task
                    or report["policy_rollouts"] != 0 or report["episodes"] != plan[stage]["episodes_per_index"]):
                raise ValueError("Predecessor cohort is incomplete or changed")
            destination = study / "cohorts" / f"s{index}" / task
            shutil.copytree(source, destination)
            if sha256(destination / "preflight.json") != expected:
                raise ValueError("Inherited cohort readback mismatch")
            files.append({"task": task, "seed_index": index, "preflight_sha256": expected})
    return {"previous_study": str(previous), "previous_plan_sha256": sha256(previous / "plan.json"),
            "previous_status_sha256": sha256(previous / "status.json"), "cohorts": files,
            "policy_results_reused": False, "selection_index_42_already_consumed": True,
            "reason": "Honor frozen expert eligibility; do not re-screen it during policy execution"}


def initialize(study, project, source_repo, inherit_cohorts_from=None):
    if study.exists():
        raise FileExistsError(f"Study already exists: {study}")
    inventory = read_json(project / "outputs/recap_50task_causal_v1/baseline_inventory.json")
    tasks = sorted(row["task"] for row in inventory["tasks"])
    if len(tasks) != 50 or len(set(tasks)) != 50:
        raise ValueError("The complete 50-task inventory is required")
    study.mkdir(parents=True)
    code = study / "code"
    shutil.copytree(source_repo, code, symlinks=True, ignore=shutil.ignore_patterns(
        ".git", ".remote_audit", ".pytest_cache", ".ruff_cache", "__pycache__", "*.egg-info"))
    wrapper = study / "model"
    hf = wrapper / "checkpoints/global_step_50000/hf_ckpt"
    hf.parent.mkdir(parents=True)
    base = Path("/dev/shm/official50k-verified-20260904")
    hf.symlink_to(base, target_is_directory=True)
    shutil.copy2("/dev/shm/recap_50task_validation_wrapper_v1/lingbotvla_cli.yaml", wrapper / "lingbotvla_cli.yaml")
    candidates = {}
    for task in TARGETS:
        artifacts = {}
        names = (["v22"] if task == "click_bell" else []) + ["positive", "signed", "regularized"]
        for name in names:
            src = (Path("/dev/shm/recap_click_bell_signed_causal_v22_adapter/recap_adapter.safetensors")
                   if name == "v22" else Path("/dev/shm/recap_50task_causal_v2/training_v2") / task / name / "adapter/recap_adapter.safetensors")
            sidecar = read_json(str(src) + ".json")
            if (sidecar["artifact_sha256"] != sha256(src)
                    or sidecar["verification"]["verified_base_tensors"] != 1708
                    or sidecar["signed_velocity_axis"] is not True):
                raise ValueError(f"Unverified artifact: {src}")
            dst = study / "artifacts" / task / f"{name}.safetensors"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            shutil.copy2(str(src) + ".json", str(dst) + ".json")
            artifacts[name] = {"path": str(dst), "sha256": sha256(dst)}
        candidates[task] = []
        for window in ("all", "d1"):
            for name in names:
                candidates[task].append({"name": f"{name}_{window}", **artifacts[name],
                                         "condition_start_decision": 0 if window == "all" else 1,
                                         "condition_decisions": -1 if window == "all" else 1})
    target_gpu = {task: 4 + index for index, task in enumerate(TARGETS)}
    task_gpu = {task: 4 + index % 4 for index, task in enumerate(tasks)}
    task_gpu.update(target_gpu)
    plan = {
        "schema_version": 1, "created_at_unix": time.time(), "project": str(project), "study": str(study),
        "all_tasks": tasks, "target_tasks": TARGETS, "candidates": candidates, "protocol": PROTOCOL,
        "gpus": [4, 5, 6, 7], "task_gpu": task_gpu, "target_gpu": target_gpu,
        "control": {"seed_indices": [32], "episodes_per_index": 3},
        "screen": {"seed_indices": [42], "episodes_per_index": 20},
        "confirm": {"seed_indices": [43, 44], "episodes_per_index": 50},
        "final": {"seed_indices": [50, 51, 52], "episodes_per_index": 20},
        "confirmation_family": TARGETS, "confirmation_holm_alpha": 0.05, "confirmation_min_delta": 0.05,
        "bootstrap_replicates": 20000, "bootstrap_seed": 20260908,
        "inference_env": "/dev/shm/conda-lingbot-recap", "sim_env": "/dev/shm/conda-robotwin",
        "conda_sh": "/root/miniconda3/etc/profile.d/conda.sh",
        "eval_workdir": str(project / "RoboTwin-E0-stable_2.0"), "model_path": str(hf),
        "base_checkpoint": str(base), "base_files": {str(p): sha256(p) for p in sorted(base.iterdir()) if p.is_file()},
        "code_sha256": code_hash(code),
        "wrapper_sha256": sha256(wrapper / "lingbotvla_cli.yaml"),
        "preregistration_sha256": sha256(code / "docs/recap_confirmatory_protocol.md"),
        "scope": "fixed_50_task_macro_improvement_not_every_task_improvement",
        "maximum_policy_episodes_excluding_infra_retries": 8064,
    }
    if inherit_cohorts_from is not None:
        plan["cohort_inheritance"] = inherit_cohorts(study, inherit_cohorts_from, plan)
    write_json(study / "plan.json", plan, immutable=True)
    write_json(study / "plan_lock.json", {"sha256": sha256(study / "plan.json")}, immutable=True)
    write_json(study / "power.json", power_sensitivity(), immutable=True)
    write_json(study / "registries/empty.json", {"schema_version": 1, "tasks": {}}, immutable=True)
    write_json(study / "inventory.json", inventory, immutable=True)
    return plan


class Study:
    def __init__(self, root):
        self.root = root
        self.plan = read_json(root / "plan.json")
        self.code = root / "code"
        self.public = Path(self.plan["project"]) / "outputs" / root.name
        self.public.mkdir(parents=True, exist_ok=True)
        self.copy_lock = threading.Lock()
        self.jobs_lock = threading.Lock()
        self.children = set()
        self.stop_requested = threading.Event()
        self.empty = root / "registries/empty.json"
        self.env = dict(os.environ)
        for key in list(self.env):
            if key.startswith("RECAP_"):
                self.env.pop(key)
        self.env.update(PYTHONNOUSERSITE="1", PYTHONHASHSEED="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                        CUBLAS_WORKSPACE_CONFIG=":4096:8", NVIDIA_TF32_OVERRIDE="0", TOKENIZERS_PARALLELISM="false",
                        HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                        PYTHONPATH=str(self.code), HF_HOME=str(Path(self.plan["project"]) / ".hf_cache"),
                        QWEN3VL_PATH=str(Path(self.plan["project"]) / "models/Qwen3-VL-4B-Instruct"))

    def verify(self):
        if sha256(self.root / "plan.json") != read_json(self.root / "plan_lock.json")["sha256"]:
            raise ValueError("Study plan changed")
        if code_hash(self.code) != self.plan["code_sha256"]:
            raise ValueError("Frozen executable source changed")
        if sha256(self.root / "model/lingbotvla_cli.yaml") != self.plan["wrapper_sha256"]:
            raise ValueError("Frozen inference YAML changed")
        if sha256(self.code / "docs/recap_confirmatory_protocol.md") != self.plan["preregistration_sha256"]:
            raise ValueError("Preregistration changed")
        for path, expected in self.plan["base_files"].items():
            if sha256(path) != expected:
                raise ValueError(f"Official base checkpoint changed: {path}")
        for candidates in self.plan["candidates"].values():
            for entry in candidates:
                if sha256(entry["path"]) != entry["sha256"]:
                    raise ValueError("Frozen adapter changed")
        if shutil.disk_usage(self.root).free < 30 * 1024**3:
            raise RuntimeError("Less than 30 GiB local staging space remains")

    def persist(self, files):
        with self.copy_lock:
            for src in files:
                dst = self.public / src.relative_to(self.root)
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(dst.name + ".tmp")
                digest = sha256(src)
                with src.open("rb") as f, tmp.open("wb") as g:
                    shutil.copyfileobj(f, g)
                    g.flush()
                    os.fsync(g.fileno())
                if sha256(tmp) != digest:
                    raise IOError(f"NFS staging mismatch: {tmp}")
                tmp.replace(dst)
                if sha256(dst) != digest:
                    raise IOError(f"NFS readback mismatch: {dst}")

    def status(self, stage, state="running", **extra):
        write_json(self.root / "status.json", {"stage": stage, "state": state, "updated_at_unix": time.time(),
                                               "pid": os.getpid(), "final_seeds_opened": (self.root / "final_opened.json").exists(), **extra})
        self.persist([self.root / "status.json"])
        print(f"STUDY {stage}: {state}", flush=True)

    def execute(self, command, *, cwd, env, log):
        if self.stop_requested.is_set():
            raise concurrent.futures.CancelledError("Study stopped before launch")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab") as handle:
            child = subprocess.Popen(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            with self.jobs_lock:
                self.children.add(child)
            try:
                return child.wait()
            finally:
                with self.jobs_lock:
                    self.children.discard(child)

    def parallel(self, jobs):
        # Report the FIRST failure immediately, not after earlier/other workers
        # finish. In-flight jobs may drain; no worker may launch another job.
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(fn, *args): index for index, (fn, args) in enumerate(jobs)}
            results = [None] * len(jobs)
            try:
                for future in concurrent.futures.as_completed(futures):
                    results[futures[future]] = future.result()
            except BaseException as error:
                self.stop_requested.set()
                for future in futures:
                    future.cancel()
                self.status("stopping", "infrastructure_or_protocol_failure_waiting_for_workers", error=repr(error))
                raise
            return results

    def sync_sim(self):
        sim = Path(self.plan["eval_workdir"]) / "script"
        (sim / "deploy").mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.code / "experiment/robotwin/eval_policy_client_lingbotvla.py", sim / "eval_policy_client_lingbotvla.py")
        for name in ("__init__.py", "websocket_client_policy.py", "msgpack_numpy.py", "recap_rollout_recorder.py",
                     "recap_instructions.py", "recap_seed_preflight.py", "recap_locked_cohort.py", "recap_evaluator_context.py"):
            shutil.copy2(self.code / "deploy" / name, sim / "deploy" / name)

    def cohort(self, task, index, count, gpu):
        folder = self.root / "cohorts" / f"s{index}" / task
        path = folder / "preflight.json"
        if not path.exists():
            folder.mkdir(parents=True, exist_ok=True)
            command = [str(Path(self.plan["sim_env"]) / "bin/python"), "-u", "script/eval_policy_client_lingbotvla.py",
                       "--config", "policy/ACT/deploy_policy.yml", "--overrides", "--task_name", task,
                       "--task_config", "demo_clean", "--seed", str(index), "--test_num", str(count),
                       "--policy_name", "ACT", "--eval_video_log", "False", "--output_dir", str(folder / "unused_policy_results"),
                       "--recap_seed_preflight", str(path), "--recap_preflight_max_candidates", str(20 * count),
                       "--recap_preflight_setup_retries", "2"]
            env = {**self.env, "CUDA_VISIBLE_DEVICES": str(gpu)}
            shell = (f"source {shlex.quote(self.plan['conda_sh'])} && conda activate {shlex.quote(self.plan['sim_env'])} && exec "
                     + shlex.join(command))
            rc = self.execute(["bash", "-c", shell], cwd=self.plan["eval_workdir"], env=env, log=folder / "preflight.log")
            if rc:
                raise RuntimeError(f"Preflight failed: {task}/s{index}, rc={rc}; see {folder}")
        report = read_json(path)
        seeds = [r["seed"] for r in report["accepted"]]
        if (report["policy_rollouts"] != 0 or report["task"] != task or report["episodes"] != count
                or len(seeds) != count or len(set(seeds)) != count
                or any(seed // 100000 - 1 != index for seed in seeds)):
            raise ValueError("Invalid preflight cohort")
        write_json(folder / "lock.json", {"sha256": sha256(path)}, immutable=True)
        self.persist([path, folder / "lock.json"])
        return report

    def cohorts(self, tasks, index, count, assignment):
        # One simulation per GPU; never concurrently place two cohort jobs on a GPU.
        def worker(gpu):
            for task in tasks:
                if assignment[task] == gpu:
                    self.cohort(task, index, count, gpu)
        self.parallel([(worker, (gpu,)) for gpu in self.plan["gpus"]])

    def registry(self, name, entries):
        path = self.root / "registries" / f"{name}.json"
        write_json(path, {"schema_version": 1, "tasks": entries}, immutable=True)
        return path

    def read_job(self, folder):
        spec = read_json(folder / "job.json")
        registry = read_json(spec["registry_path"])
        if sha256(spec["registry_path"]) != spec["registry_sha256"]:
            raise ValueError("Job registry changed")
        if sha256(folder / "cohorts.json") != spec["cohort_manifest_sha256"]:
            raise ValueError("Job cohort manifest changed")
        result = {}
        for task in spec["tasks"]:
            cohort = load_locked_cohort(
                folder / "cohorts.json", task=task, task_config=self.plan["protocol"]["task_config"],
                seeds=spec["seeds"][task], count=spec["count"],
                instructions=[spec["instructions"][task][str(seed)] for seed in spec["seeds"][task]],
            )
            attempts = [p for p in (folder / "rollouts").glob("attempt-*") if (p / task).is_dir()]
            if not attempts:
                raise ValueError(f"No completed rollout directory: {folder}/{task}")
            latest = max(attempts, key=lambda p: int(p.name.split("-")[-1]))
            rows = {}
            for path in sorted((latest / task).rglob("manifest.json")):
                row = read_json(path)
                seed = row["seed"]
                if seed in rows or str(seed) not in spec["instructions"][task]:
                    raise ValueError("Duplicate/unplanned episode")
                validate_runtime(row, task=task, seed=seed, instruction=spec["instructions"][task][str(seed)],
                                 condition=spec["condition"], registry=registry, protocol=self.plan["protocol"], cohort=cohort)
                if recorded_initial_hashes(path, row) != cohort.entries[seed]["initial_observation_sha256"]:
                    raise ValueError("Recorded initial arrays differ from the locked preflight")
                rows[seed] = (path, row)
            if set(rows) != set(spec["seeds"][task]):
                raise ValueError(f"Incomplete planned cohort: {folder}/{task}: {len(rows)}/{spec['count']}")
            result[task] = rows
        done = folder / "done.json"
        if done.exists():
            for record in read_json(done)["files"]:
                if sha256(folder / record["path"]) != record["sha256"]:
                    raise ValueError(f"Completed rollout evidence changed: {record['path']}")
        return result

    def read_job_before_completion(self, folder):
        try:
            return self.read_job(folder)
        except Exception as error:
            # The launcher has returned: preserve failed attempts too, even
            # though no complete-cohort done.json can be issued.
            failure = folder / "failure.json"
            if not failure.exists():
                write_json(failure, {"error": repr(error), "created_at_unix": time.time()}, immutable=True)
            self.persist([p for p in sorted(folder.rglob("*")) if p.is_file()])
            raise

    def persist_job(self, folder):
        if not (folder / "persisted.json").exists():
            self.persist([p for p in sorted(folder.rglob("*")) if p.is_file()])
            write_json(folder / "persisted.json", {"done_sha256": sha256(folder / "done.json")}, immutable=True)
            self.persist([folder / "persisted.json"])

    def evaluate(self, phase, index, variant, tasks, count, gpu, registry, condition):
        if self.stop_requested.is_set():
            raise concurrent.futures.CancelledError("Study stopped before evaluation")
        folder = self.root / "evaluations" / phase / f"s{index}" / variant / f"gpu{gpu}"
        seeds, instructions, cohorts = {}, {}, {}
        for task in tasks:
            report = read_json(self.root / "cohorts" / f"s{index}" / task / "preflight.json")
            if report["episodes"] != count:
                raise ValueError("Cohort size changed")
            seeds[task] = [row["seed"] for row in report["accepted"]]
            instructions[task] = {str(row["seed"]): row["instruction"] for row in report["accepted"]}
            path = self.root / "cohorts" / f"s{index}" / task / "preflight.json"
            expected = read_json(path.with_name("lock.json"))["sha256"]
            if sha256(path) != expected:
                raise ValueError("Frozen cohort changed before evaluation")
            cohorts[task] = {"path": str(path), "sha256": expected}
        write_json(folder / "cohorts.json", {"schema_version": 1, "task_config": self.plan["protocol"]["task_config"],
                                            "tasks": cohorts}, immutable=True)
        spec = {"phase": phase, "seed_index": index, "variant": variant, "tasks": tasks, "count": count,
                "gpu": gpu, "registry_path": str(registry), "registry_sha256": sha256(registry),
                "condition": condition, "seeds": seeds, "instructions": instructions,
                "code_sha256": self.plan["code_sha256"],
                "cohort_manifest_sha256": sha256(folder / "cohorts.json")}
        write_json(folder / "job.json", spec, immutable=True)
        write_json(folder / "seeds.json", {"schema_version": 1, "tasks": seeds}, immutable=True)
        write_json(folder / "instructions.json", {"schema_version": 1, "tasks": instructions}, immutable=True)
        if (folder / "done.json").exists():
            rows = self.read_job(folder)
            self.persist_job(folder)
            return rows
        if (folder / "launch.json").exists():
            # A terminated supervisor may leave completed authoritative manifests.
            # Do not silently run a different trial under the same statistical ID.
            rows = self.read_job_before_completion(folder)
            rc = "recovered_from_complete_local_manifests"
        else:
            write_json(folder / "launch.json", {"started_at_unix": time.time()}, immutable=True)
            command = ["bash", str(self.code / "experiment/robotwin/start_robotwin_infer_and_eval.sh"),
                       "--model_path", self.plan["model_path"], "--inference_workdir", str(self.code) + "/",
                       "--eval_workdir", self.plan["eval_workdir"], "--output_base", str(folder / "eval"),
                       "--conda_sh", self.plan["conda_sh"], "--inference_env", self.plan["inference_env"],
                       "--sim_env", self.plan["sim_env"], "--tasks", ",".join(tasks), "--test_num", str(count),
                       "--seed", str(index), "--policy_seed", "900", "--continuation_policy_seed", "300",
                       "--counterfactual_policy_decision", "0", "--common_noise_per_episode",
                       "--recap_environment_seed_map", str(folder / "seeds.json"), "--recap_setup_retries", "5",
                       "--recap_cohort_manifest", str(folder / "cohorts.json"),
                       "--recap_instruction_map", str(folder / "instructions.json"), "--recap_adapter_registry", str(registry),
                       "--recap_condition", condition, "--num_gpus", "1", "--gpu_offset", str(gpu),
                       "--start_port", str(10100 + gpu), "--use_length", "50", "--no_video",
                       "--use_bf16", "False", "--use_fp32", "True", "--use_compile", "False",
                       "--deterministic_algorithms", "True", "--recap_rollout_dir", str(folder / "rollouts")]
            rc = self.execute(command, cwd=self.code, env=self.env, log=folder / "launcher.log")
            rows = self.read_job_before_completion(folder)  # Local manifests, not NFS result JSON, are authoritative.
        manifests = [{"task": task, "seed": seed, "path": str(path), "sha256": sha256(path)}
                     for task, values in rows.items() for seed, (path, _) in values.items()]
        files = []
        for values in rows.values():
            for path, row in values.values():
                for artifact in [path, *(path.parent / step["file"] for step in row["steps"])]:
                    files.append({"path": str(artifact.relative_to(folder)), "sha256": sha256(artifact)})
        write_json(folder / "done.json", {"launcher_status": rc, "manifests": manifests, "files": files}, immutable=True)
        self.persist_job(folder)
        return rows

    def batch(self, phase, index, variant, tasks, count, assignment, registry, condition):
        jobs = [(self.evaluate, (phase, index, variant, [t for t in tasks if assignment[t] == gpu],
                                 count, gpu, registry, condition)) for gpu in self.plan["gpus"]
                if any(assignment[t] == gpu for t in tasks)]
        result = {}
        for rows in self.parallel(jobs):
            result.update(rows)
        return result

    def paired(self, positive, null):
        if positive.keys() != null.keys():
            raise ValueError("Task sets differ")
        results = {}
        for task in positive:
            if positive[task].keys() != null[task].keys():
                raise ValueError("Seed sets differ")
            for seed in positive[task]:
                if positive[task][seed][1]["task"] != null[task][seed][1]["task"]:
                    raise ValueError("Paired instruction mismatch")
                drift = initial_drift(positive[task][seed], null[task][seed])
                if drift["numeric_max_abs"] > 0.001 or drift["image_mae_max"] > 2.0:
                    raise ValueError(f"Initial state mismatch: {task}/{seed}: {drift}")
            seeds = sorted(positive[task])
            results[task] = paired_counts([positive[task][s][1]["success"] for s in seeds],
                                         [null[task][s][1]["success"] for s in seeds])
        return results

    def run(self, stop_after=None):
        self.verify()
        self.sync_sim()
        archive = self.root / "frozen_inputs.tar.gz"
        if not (self.root / "frozen_inputs.json").exists():
            temporary = archive.with_suffix(".tmp")
            with tarfile.open(temporary, "w:gz") as output:
                for name in ("code", "artifacts", "model/lingbotvla_cli.yaml", "plan.json", "plan_lock.json", "power.json"):
                    output.add(self.root / name, arcname=name)
            temporary.replace(archive)
            write_json(self.root / "frozen_inputs.json", {"sha256": sha256(archive)}, immutable=True)
        if sha256(archive) != read_json(self.root / "frozen_inputs.json")["sha256"]:
            raise ValueError("Frozen input archive changed")
        self.persist([archive, self.root / "frozen_inputs.json", self.root / "plan.json",
                      self.root / "plan_lock.json", self.root / "power.json", self.code / "docs/recap_confirmatory_protocol.md"])
        tasks, assignment = self.plan["target_tasks"], self.plan["target_gpu"]
        self.status("control")
        self.cohorts(tasks, 32, 3, assignment)
        null = self.batch("control", 32, "null", tasks, 3, assignment, self.empty, "null")
        fallback = self.batch("control", 32, "fallback", tasks, 3, assignment, self.empty, "positive")
        control = self.paired(fallback, null)
        if any(row["improved"] or row["regressed"] for row in control.values()):
            raise RuntimeError(f"Empty-registry control failed: {control}")
        write_json(self.root / "control.json", control, immutable=True)
        self.persist([self.root / "control.json"])
        if stop_after == "control":
            self.status("control", "control_passed_awaiting_resume")
            return

        self.verify()
        self.status("screen_42")
        self.cohorts(tasks, 42, 20, assignment)
        null = self.batch("screen", 42, "null", tasks, 20, assignment, self.empty, "null")
        def screen_task(task):
            metrics = []
            for priority, entry in enumerate(self.plan["candidates"][task]):
                registry = self.registry(f"screen_{task}_{entry['name']}", {task: entry})
                rows = self.evaluate("screen", 42, entry["name"], [task], 20, assignment[task], registry, "positive")
                stat = self.paired(rows, {task: null[task]})[task]
                metrics.append({"candidate": entry, "priority": priority, **stat})
            chosen = max(metrics, key=lambda row: (row["improved"] - row["regressed"], -row["regressed"], -row["priority"]))
            return task, {"chosen": chosen["candidate"], "metrics": metrics}
        screen = dict(self.parallel([(screen_task, (task,)) for task in tasks]))
        write_json(self.root / "screen_selection.json", screen, immutable=True)
        shortlist = self.registry("shortlist", {task: row["chosen"] for task, row in screen.items()})
        write_json(self.root / "shortlist_lock.json", {"sha256": sha256(shortlist)}, immutable=True)
        self.persist([self.root / "screen_selection.json", shortlist, self.root / "shortlist_lock.json"])
        if stop_after == "screen":
            self.status("screen", "shortlist_frozen_awaiting_resume")
            return

        self.verify()
        self.status("confirm_43_44")
        confirmation = {condition: {task: {} for task in tasks} for condition in ("positive", "null", "negative")}
        for index in (43, 44):
            self.cohorts(tasks, index, 50, assignment)
            for condition in (("null", "positive", "negative") if index == 43 else ("negative", "positive", "null")):
                rows = self.batch("confirm", index, condition, tasks, 50, assignment,
                                  self.empty if condition == "null" else shortlist, condition)
                for task in tasks:
                    confirmation[condition][task].update(rows[task])
        stats = self.paired(confirmation["positive"], confirmation["null"])
        negative_stats = self.paired(confirmation["negative"], confirmation["null"])
        adjusted = holm_adjust({task: row["two_sided_p"] for task, row in stats.items()})
        selected = {}
        for task, row in stats.items():
            row["holm_p"] = adjusted[task]
            row["negative_success"] = negative_stats[task]["positive_success"]
            row["passed"] = bool(row["positive_success"] > row["null_success"] > row["negative_success"]
                                 and row["delta"] >= 0.05 and row["holm_p"] < 0.05)
            if row["passed"]:
                selected[task] = screen[task]["chosen"]
        write_json(self.root / "confirmation.json", stats, immutable=True)
        self.persist([self.root / "confirmation.json"])
        if not selected:
            self.status("confirmation", "no_validated_candidate", final_seeds_preserved=True)
            return
        final_registry = self.registry("final", selected)
        write_json(self.root / "registry_freeze.json", {"registry_sha256": sha256(final_registry),
                   "confirmation_sha256": sha256(self.root / "confirmation.json"),
                   "plan_sha256": sha256(self.root / "plan.json")}, immutable=True)
        self.persist([final_registry, self.root / "registry_freeze.json"])
        if stop_after == "confirm":
            self.status("confirmation", "registry_frozen_awaiting_resume")
            return

        self.verify()
        write_json(self.root / "final_opened.json", {"registry_sha256": sha256(final_registry)}, immutable=True)
        self.status("final_50_52")
        all_tasks, fixed_gpu = self.plan["all_tasks"], self.plan["task_gpu"]
        final = {condition: {task: {} for task in all_tasks} for condition in ("positive", "null")}
        negatives = {task: {} for task in selected}
        for index in (50, 51, 52):
            self.cohorts(all_tasks, index, 20, fixed_gpu)
            for condition in (("null", "positive") if index != 51 else ("positive", "null")):
                rows = self.batch("final", index, condition, all_tasks, 20, fixed_gpu,
                                  self.empty if condition == "null" else final_registry, condition)
                for task in all_tasks:
                    final[condition][task].update(rows[task])
            rows = self.batch("final", index, "negative", sorted(selected), 20, fixed_gpu, final_registry, "negative")
            for task in selected:
                negatives[task].update(rows[task])
        self.verify()
        if sha256(final_registry) != read_json(self.root / "registry_freeze.json")["registry_sha256"]:
            raise ValueError("Final registry changed")
        self.paired(final["positive"], final["null"])
        pairs = {task: [{"seed_index": seed // 100000 - 1, "seed": seed,
                         "positive": final["positive"][task][seed][1]["success"],
                         "null": final["null"][task][seed][1]["success"]}
                        for seed in sorted(final["null"][task])] for task in all_tasks}
        report = final_statistics(pairs, tasks=all_tasks, registry_tasks=selected)
        report["secondary_negative"] = self.paired(negatives, {task: final["null"][task] for task in selected})
        report["registry_sha256"] = sha256(final_registry)
        report["plan_sha256"] = sha256(self.root / "plan.json")
        write_json(self.root / "final_pairs.json", pairs, immutable=True)
        write_json(self.root / "final_result.json", report, immutable=True)
        self.persist([self.root / "final_pairs.json", self.root / "final_result.json"])
        self.status("complete", "significant_uplift" if report["passed"] else "not_proven",
                    macro_delta=report["macro_delta"], p=report["two_sided_exact_p"], ci=report["paired_bootstrap_95ci"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--stop-after", choices=["control", "screen", "confirm"])
    parser.add_argument("--project", type=Path, default=Path("/vla-cd/ReconVLA/Lingbot-VLA"))
    parser.add_argument("--source-repo", type=Path, default=Path("/dev/shm/lingbot-vla-v2-recap"))
    parser.add_argument("--inherit-cohorts-from", type=Path,
                        help="New amended snapshot only: preserve complete consumed-index 32/42 cohorts, not outcomes")
    args = parser.parse_args()
    if args.inherit_cohorts_from is not None and not args.initialize:
        parser.error("--inherit-cohorts-from requires --initialize of a NEW study")
    if args.initialize:
        initialize(args.study, args.project, args.source_repo, args.inherit_cohorts_from)
    frozen_script = args.study / "code/scripts/recap_confirmatory_study.py"
    if Path(__file__).resolve() != frozen_script.resolve():
        argv = [sys.executable, str(frozen_script), "--study", str(args.study)]
        if args.stop_after:
            argv += ["--stop-after", args.stop_after]
        os.execve(sys.executable, argv, {**os.environ, "PYTHONPATH": str(args.study / "code")})
    with (args.study / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study = Study(args.study)
        try:
            study.run(stop_after=args.stop_after)
        except BaseException as error:
            for child in list(study.children):
                child.terminate()
            try:
                study.status("stopped", "infrastructure_or_protocol_failure", error=repr(error))
            finally:
                raise


if __name__ == "__main__":
    main()
