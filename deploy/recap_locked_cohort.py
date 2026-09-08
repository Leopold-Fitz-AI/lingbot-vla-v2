"""Execute an already accepted cohort without reopening expert eligibility.

This is NOT a generic skip-expert option: exact seeds, literal instructions,
preflight evidence and its checksum lock must agree before policy execution.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from deploy.recap_evaluator_context import validate_evaluator_context
from deploy.recap_instructions import INSTRUCTION_PROTOCOL
from deploy.recap_seed_preflight import initial_observation_hashes


LOCKED_COHORT_PROTOCOL = "locked-expert-preflight-v1"
PROTOCOL_ERROR_EXIT = 78


class CohortProtocolError(ValueError):
    """An immutable input/state mismatch; never retry to get matching data."""


@dataclass(frozen=True)
class LockedCohort:
    path: Path
    sha256: str
    entries: dict

    def initial_metadata(self, seed, observation):
        actual = initial_observation_hashes(observation)
        if actual != self.entries[seed]["initial_observation_sha256"]:
            # Do not retry until a favorable initialization or silently resample.
            raise CohortProtocolError(f"Locked preflight initial observation mismatch: seed={seed}")
        return {"eligibility_protocol": LOCKED_COHORT_PROTOCOL,
                "preflight_sha256": self.sha256,
                "initial_observation_sha256": actual}


def load_locked_cohort(manifest, *, task, task_config, seeds, instructions, count):
    """Validate the whole planned task cohort, not just the current episode."""
    try:
        return _load_locked_cohort(manifest, task, task_config, seeds, instructions, count)
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise CohortProtocolError(str(error)) from error


def _load_locked_cohort(manifest, task, task_config, seeds, instructions, count):
    mapping = json.loads(Path(manifest).read_bytes())
    if (mapping.get("schema_version") != 1 or mapping.get("task_config") != task_config
            or not isinstance(mapping.get("tasks"), dict) or task not in mapping["tasks"]):
        raise ValueError("Invalid locked cohort manifest/task/config")
    entry = mapping["tasks"][task]
    path = Path(entry["path"])
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    lock = json.loads(path.with_name("lock.json").read_bytes())
    if digest != entry.get("sha256") or digest != lock.get("sha256"):
        raise ValueError("Locked preflight checksum mismatch")
    report = json.loads(raw)
    if (report.get("schema_version") != 1
            or report.get("purpose") != "expert_feasibility_only_no_learned_policy"
            or report.get("policy_rollouts") != 0 or report.get("task") != task
            or report.get("instruction_protocol") != INSTRUCTION_PROTOCOL
            or report.get("episodes") != count):
        raise ValueError("Invalid expert-only preflight evidence")
    accepted = report["accepted"]
    expected_seeds = [row["seed"] for row in accepted]
    if (not isinstance(seeds, list) or not seeds or len(seeds) != count
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != count or seeds != expected_seeds):
        raise ValueError("Locked cohort requires the entire exact ordered seed map")
    if ([row["seed"] for row in report["attempts"] if row.get("accepted") is True]
            != expected_seeds):
        raise ValueError("Missing accepted expert attempt evidence")
    expected_text = [row["instruction"] for row in accepted]
    if instructions != expected_text or any(not isinstance(s, str) or not s.strip() for s in expected_text):
        raise ValueError("Locked cohort requires exact literal instructions")
    for row in accepted:
        validate_evaluator_context(task, row.get("evaluator_context", {}))
        hashes = row["initial_observation_sha256"]
        if (set(hashes) != {"state", "head_camera", "left_camera", "right_camera"}
                or any(not isinstance(h, str) or len(h) != 64
                       or any(c not in "0123456789abcdef" for c in h) for h in hashes.values())):
            raise ValueError("Missing preflight initial observation evidence")
    return LockedCohort(path, digest, {row["seed"]: row for row in accepted})
