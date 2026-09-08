"""Restore evaluator-only fields that three RoboTwin tasks set in play_once.

Never execute or reimplement a success predicate. Copy only this allowlist from
an accepted expert preflight, preserving scalar dtype and native ArmTag type.
"""

import importlib

import numpy as np


CONTEXT_FIELDS = {
    "open_laptop": {"arm_tag"},
    "place_object_scale": {"arm_tag"},
    "put_object_cabinet": {"arm_tag", "origin_z"},
}


def validate_evaluator_context(task, context):
    if not isinstance(context, dict) or set(context) != CONTEXT_FIELDS.get(task, set()):
        raise ValueError(f"Missing/unexpected evaluator context for {task}")
    if "arm_tag" in context and context["arm_tag"] not in ("left", "right"):
        raise ValueError("Invalid evaluator arm tag")
    if "origin_z" in context:
        value = context["origin_z"]
        if (not isinstance(value, dict) or set(value) != {"kind", "dtype", "value"}
                or value["kind"] not in ("numpy_scalar", "python_float")
                or value["dtype"] not in ("float32", "float64")
                or not np.isfinite(value["value"])):
            raise ValueError("Invalid evaluator origin_z")
    return context


def capture_evaluator_context(env, task):
    context = {}
    if task in CONTEXT_FIELDS:
        context["arm_tag"] = str(env.arm_tag)
    if task == "put_object_cabinet":
        value = env.origin_z
        if not isinstance(value, (float, np.floating)):
            raise ValueError("Unexpected evaluator origin_z type")
        context["origin_z"] = {"kind": "numpy_scalar" if isinstance(value, np.floating) else "python_float",
                               "dtype": str(np.asarray(value).dtype), "value": float(value)}
    return validate_evaluator_context(task, context)


def restore_evaluator_context(env, task, context):
    validate_evaluator_context(task, context)
    if "arm_tag" in context:
        # The installed task module already imports RoboTwin's native ArmTag.
        arm_type = importlib.import_module(type(env).__module__).ArmTag
        env.arm_tag = arm_type(context["arm_tag"])
    if "origin_z" in context:
        value = context["origin_z"]
        env.origin_z = (np.dtype(value["dtype"]).type(value["value"]) if value["kind"] == "numpy_scalar"
                        else float(value["value"]))
    if capture_evaluator_context(env, task) != context:
        raise ValueError("Evaluator context restore mismatch")
