import hashlib
import json
import sys
import os
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from deploy.recap_instructions import INSTRUCTION_PROTOCOL, generate_task_instruction
from deploy.recap_locked_cohort import CohortProtocolError, PROTOCOL_ERROR_EXIT
from deploy.recap_evaluator_context import capture_evaluator_context, restore_evaluator_context
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
_INSTRUCTION_MAP_CACHE = {}
_DECISION_MAP_CACHE = {}
_ENVIRONMENT_SEED_MAP_CACHE = {}


def verified_atomic_write_json(path, payload):
    """Write JSON through local memory and verify destination bytes."""
    path = Path(path)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    token = hashlib.sha256(f"{path}:{os.getpid()}".encode()).hexdigest()[:12]
    local = Path("/dev/shm") / f".recap-result-{token}.tmp"
    destination = path.with_name(f".{path.name}.{token}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with local.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if hashlib.sha256(local.read_bytes()).hexdigest() != digest:
            raise IOError(f"Local JSON staging verification failed: {local}")
        with destination.open("wb") as handle:
            handle.write(local.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise IOError(f"Destination JSON staging verification failed: {destination}")
        os.replace(destination, path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise IOError(f"Final JSON verification failed: {path}")
    finally:
        local.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)


def deterministic_instructions_enabled():
    return os.environ.get("RECAP_DETERMINISTIC_INSTRUCTIONS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def mapped_counterfactual_decision(task_name):
    path = os.environ.get(
        "RECAP_COUNTERFACTUAL_POLICY_DECISION_MAP", ""
    ).strip()
    if not path:
        raw = os.environ.get("RECAP_COUNTERFACTUAL_POLICY_DECISION")
        return None if raw in (None, "") else int(raw)
    if path not in _DECISION_MAP_CACHE:
        with open(path) as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != 1 or not isinstance(
            payload.get("tasks"), dict
        ):
            raise ValueError(
                "Counterfactual decision map must have schema_version=1 and a tasks object"
            )
        _DECISION_MAP_CACHE[path] = payload["tasks"]
    if task_name not in _DECISION_MAP_CACHE[path]:
        raise ValueError(f"Decision map has no entry for task={task_name!r}")
    return int(_DECISION_MAP_CACHE[path][task_name])


def mapped_environment_seeds(task_name):
    path = os.environ.get("RECAP_ENVIRONMENT_SEED_MAP", "").strip()
    if not path:
        return None
    if path not in _ENVIRONMENT_SEED_MAP_CACHE:
        with open(path) as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != 1 or not isinstance(
            payload.get("tasks"), dict
        ):
            raise ValueError(
                "Environment seed map must have schema_version=1 and a tasks object"
            )
        _ENVIRONMENT_SEED_MAP_CACHE[path] = payload["tasks"]
    seeds = _ENVIRONMENT_SEED_MAP_CACHE[path].get(task_name)
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"Environment seed map has no seeds for task={task_name!r}")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        for seed in seeds
    ):
        raise ValueError(f"Environment seed map for task={task_name!r} is invalid")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Environment seed map for task={task_name!r} has duplicates")
    return list(seeds)


def mapped_task_instruction(task_name, seed):
    path = os.environ.get("RECAP_TASK_INSTRUCTION_MAP", "").strip()
    if not path:
        return None
    if path not in _INSTRUCTION_MAP_CACHE:
        with open(path) as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != 1 or not isinstance(
            payload.get("tasks"), dict
        ):
            raise ValueError(
                "RECAP instruction map must have schema_version=1 and a tasks object"
            )
        _INSTRUCTION_MAP_CACHE[path] = payload["tasks"]
    task_instructions = _INSTRUCTION_MAP_CACHE[path].get(task_name)
    if not isinstance(task_instructions, dict) or str(seed) not in task_instructions:
        raise ValueError(
            f"Instruction map has no entry for task={task_name!r}, seed={seed}"
        )
    instruction = task_instructions[str(seed)]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(
            f"Instruction map entry for task={task_name!r}, seed={seed} is invalid"
        )
    return instruction


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None
    video_fps = str(usr_args.get("video_fps", 10))

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    if usr_args.get("output_dir"):
        save_dir = Path(usr_args["output_dir"]) / task_name
    else:
        save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 命令行 --eval_video_log 优先于 YAML 配置
    if "eval_video_log" in usr_args:
        args["eval_video_log"] = usr_args["eval_video_log"]

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = int(usr_args.get("test_num", 100))
    if test_num <= 0:
        raise ValueError(f"test_num must be positive, got {test_num}")
    topk = 1

    preflight_path = usr_args.get("recap_seed_preflight")
    if preflight_path:
        # No WebSocket connection or learned-policy outcome is involved in
        # constructing the cohort. Keep the report distinct from _result.json.
        from deploy.recap_seed_preflight import preflight_environment_seeds
        report = preflight_environment_seeds(
            TASK_ENV, args, task=task_name, start_seed=st_seed, count=test_num,
            generator=generate_episode_descriptions, instruction_type=instruction_type,
            max_candidates=int(usr_args.get("recap_preflight_max_candidates", test_num * 20)),
            setup_retries=int(usr_args.get("recap_preflight_setup_retries", 2)),
            progress=lambda record: print("PREFLIGHT " + json.dumps(record), flush=True),
        )
        verified_atomic_write_json(preflight_path, report)
        return

    # model = get_model(usr_args)
    # from IPython import embed;embed()
    from script.deploy.websocket_client_policy import WebsocketClientPolicy
    model = WebsocketClientPolicy(port=usr_args['port'])

    st_seed, suc_num, episode_outcomes = eval_policy(
        task_name,
        TASK_ENV,
        args,
        model,
        st_seed,
        test_num=test_num,
        video_size=video_size,
        video_fps=video_fps,
        instruction_type=instruction_type,
        usr_args=usr_args,
    )
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    result_payload = {
        "schema_version": 1,
        "timestamp": current_time,
        "task": task_name,
        "task_config": args["task_config"],
        "instruction_type": instruction_type,
        "seed_index": int(usr_args.get("seed", 0)),
        "success": int(suc_num),
        "episodes": int(test_num),
        "success_rate": float(suc_num / test_num),
        "episode_outcomes": episode_outcomes,
    }
    result_path = Path(save_dir) / "_result.json"
    verified_atomic_write_json(result_path, result_payload)
    recap_rollout_dir = str(usr_args.get("recap_rollout_dir", "")).strip()
    if recap_rollout_dir:
        verified_atomic_write_json(
            Path(recap_rollout_dir) / f"_{task_name}_result.json", result_payload
        )
    print(f"Data has been saved to {file_path} and {result_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                video_fps="10",
                instruction_type=None,
                usr_args = None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    recap_rollout_dir = None if usr_args is None else usr_args.get("recap_rollout_dir")
    RecapEpisodeRecorder = None
    if recap_rollout_dir:
        from script.deploy.recap_rollout_recorder import RecapEpisodeRecorder

    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    # eval_func = eval_function_decorator(policy_name, "eval")
    # reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    fixed_environment_seeds = mapped_environment_seeds(task_name)
    if fixed_environment_seeds is not None and len(fixed_environment_seeds) != test_num:
        raise ValueError(
            f"Environment seed map for task={task_name!r} contains "
            f"{len(fixed_environment_seeds)} seeds, expected test_num={test_num}"
        )
    locked_cohort = None
    cohort_manifest = os.environ.get("RECAP_COHORT_MANIFEST", "").strip()
    if cohort_manifest:
        from deploy.recap_locked_cohort import load_locked_cohort
        locked_cohort = load_locked_cohort(
            cohort_manifest, task=task_name, task_config=args["task_config"],
            seeds=fixed_environment_seeds, count=test_num,
            instructions=([mapped_task_instruction(task_name, seed) for seed in fixed_environment_seeds]
                          if fixed_environment_seeds else None),
        )
    # Ordinary benchmark runs still perform their legacy live expert check.
    # A verified locked cohort has already passed this outcome-blind gate.
    expert_check = locked_cohort is None
    setup_retries = int(os.environ.get("RECAP_SETUP_RETRIES", "3"))
    if setup_retries < 0:
        raise ValueError("RECAP_SETUP_RETRIES must be non-negative")
    setup_failures = {}
    expert_calls = {}
    episode_outcomes = []
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        if fixed_environment_seeds is not None:
            now_seed = fixed_environment_seeds[succ_seed]
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                expert_calls[now_seed] = expert_calls.get(now_seed, 0) + 1
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                TASK_ENV.close_env()
                args["render_freq"] = render_freq
                if fixed_environment_seeds is None:
                    now_seed += 1
                else:
                    setup_failures[now_seed] = setup_failures.get(now_seed, 0) + 1
                    if setup_failures[now_seed] > setup_retries:
                        raise RuntimeError(
                            f"Fixed environment seed {now_seed} remained unstable "
                            f"after {setup_retries + 1} attempts"
                        ) from e
                continue
            except Exception as e:
                TASK_ENV.close_env()
                args["render_freq"] = render_freq
                if fixed_environment_seeds is None:
                    now_seed += 1
                else:
                    setup_failures[now_seed] = setup_failures.get(now_seed, 0) + 1
                    if setup_failures[now_seed] > setup_retries:
                        raise RuntimeError(
                            f"Fixed environment seed {now_seed} failed setup "
                            f"after {setup_retries + 1} attempts"
                        ) from e
                print(f"error occurs ! {e}")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            args["render_freq"] = render_freq
            if fixed_environment_seeds is None:
                now_seed += 1
            else:
                setup_failures[now_seed] = setup_failures.get(now_seed, 0) + 1
                if setup_failures[now_seed] > setup_retries:
                    raise RuntimeError(
                        f"Fixed environment seed {now_seed} failed expert validation "
                        f"after {setup_retries + 1} attempts"
                    )
            continue

        args["render_freq"] = render_freq

        try:
            TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        except Exception as error:
            TASK_ENV.close_env()
            succ_seed -= 1
            suc_test_seed_list.pop()
            if fixed_environment_seeds is None:
                now_seed += 1
            else:
                setup_failures[now_seed] = setup_failures.get(now_seed, 0) + 1
                if setup_failures[now_seed] > setup_retries:
                    raise RuntimeError(
                        f"Fixed environment seed {now_seed} failed rollout setup "
                        f"after {setup_retries + 1} attempts"
                    ) from error
            continue
        # Locked execution uses the literal, verified preflight instruction;
        # it must not synthesize new text or call an expert to reconstruct info.
        episode_info_list = [] if locked_cohort else [episode_info["info"]]
        deterministic_instructions = deterministic_instructions_enabled()
        instruction_override = os.environ.get("RECAP_TASK_INSTRUCTION", "").strip()
        instruction_map_path = os.environ.get(
            "RECAP_TASK_INSTRUCTION_MAP", ""
        ).strip()
        configured_sources = sum(
            bool(value)
            for value in (
                instruction_override,
                instruction_map_path,
                deterministic_instructions,
            )
        )
        if configured_sources > 1:
            raise ValueError(
                "Use only one fixed, mapped, or deterministic RECAP instruction source"
            )
        instruction_from_map = (
            mapped_task_instruction(task_name, now_seed)
            if instruction_map_path
            else None
        )
        instruction = (
            instruction_override
            or instruction_from_map
            or generate_task_instruction(
                task_name,
                now_seed,
                episode_info_list,
                instruction_type,
                test_num,
                deterministic=deterministic_instructions,
                generator=generate_episode_descriptions,
            )
        )
        instruction_source = (
            "fixed"
            if instruction_override
            else "map"
            if instruction_from_map is not None
            else "deterministic"
            if deterministic_instructions
            else "random"
        )
        if locked_cohort and instruction != locked_cohort.entries[now_seed]["instruction"]:
            raise CohortProtocolError("Instruction override conflicts with locked preflight")
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        initial_observation = None
        eligibility_metadata = {"eligibility_protocol": "live_expert"}
        if locked_cohort:
            context = locked_cohort.entries[now_seed].get("evaluator_context", {})
            try:
                restore_evaluator_context(TASK_ENV, task_name, context)
            except (ValueError, AttributeError, KeyError) as error:
                raise CohortProtocolError(f"Evaluator context restore failed: {error}") from error
            initial_observation = TASK_ENV.get_obs()
            eligibility_metadata = locked_cohort.initial_metadata(now_seed, initial_observation)
            eligibility_metadata["evaluator_context"] = capture_evaluator_context(TASK_ENV, task_name)
        eligibility_metadata["expert_rollouts_before_policy"] = expert_calls.get(now_seed, 0)

        recap_recorder = None
        if RecapEpisodeRecorder is not None:
            rollout_root = Path(recap_rollout_dir) / task_name
            recap_run_id = usr_args.get("recap_run_id", "run")
            episode_id = f"{task_name}-{recap_run_id}-seed-{now_seed}-episode-{now_id}"
            recap_recorder = RecapEpisodeRecorder(
                rollout_root,
                episode_id=episode_id,
                task=str(instruction),
                seed=now_seed,
                metadata={
                    "task_name": task_name,
                    "policy_name": policy_name,
                    "task_config": args.get("task_config"),
                    "checkpoint": args.get("ckpt_setting"),
                    "recap_condition": usr_args.get("recap_condition", "positive"),
                    "recap_cfg_scale": usr_args.get("recap_cfg_scale", 1.0),
                    "policy_seed": os.environ.get("RECAP_POLICY_SEED"),
                    "continuation_policy_seed": os.environ.get(
                        "RECAP_CONTINUATION_POLICY_SEED"
                    ),
                    "counterfactual_policy_decision": mapped_counterfactual_decision(
                        task_name
                    ),
                    "counterfactual_policy_decision_map": os.environ.get(
                        "RECAP_COUNTERFACTUAL_POLICY_DECISION_MAP"
                    ),
                    "recap_condition_start_decision": os.environ.get(
                        "RECAP_CONDITION_START_DECISION"
                    ),
                    "recap_condition_decisions": os.environ.get(
                        "RECAP_CONDITION_DECISIONS"
                    ),
                    "instruction_overridden": instruction_source != "random",
                    "instruction_source": instruction_source,
                    "instruction_protocol": (
                        INSTRUCTION_PROTOCOL if deterministic_instructions else instruction_source
                    ),
                    "instruction_map": os.environ.get("RECAP_TASK_INSTRUCTION_MAP"),
                    "environment_seed_map": os.environ.get(
                        "RECAP_ENVIRONMENT_SEED_MAP"
                    ),
                    "common_noise_per_episode": os.environ.get(
                        "RECAP_COMMON_NOISE_PER_EPISODE"
                    ),
                    "step_limit": TASK_ENV.step_lim,
                    **eligibility_metadata,
                },
            )

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    video_fps,
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        path_to_pi_model = None
        if usr_args is not None and "new_ckpt_path" in usr_args:
            path_to_pi_model = usr_args['new_ckpt_path']
        ret = model.infer(
            dict(
                reset=True,
                robo_name=usr_args['robo_name'],
                path_to_pi_model=path_to_pi_model,
                task_name=task_name,
                environment_seed=now_seed,
            )
        )
        
        while TASK_ENV.take_action_cnt<TASK_ENV.step_lim and not succ:
            observation = initial_observation if initial_observation is not None else TASK_ENV.get_obs()
            initial_observation = None
            # from IPython import embed;embed()
            
            formatted_observation = {
                "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"], # H,W,3
                "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
                "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
                
                "observation.state": observation["joint_action"]["vector"],
                "task": TASK_ENV.get_instruction(),
            }

            # print(formatted_observation["observation.images.cam_high"].shape)
            # ========= debug image =========
            # import torch, torchvision
            # from PIL import Image
            # imgs_np = []
            # for k in ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']:
            #     arr = formatted_observation['image'][k].squeeze(0)  # (H,W,3), uint8
            #     print(arr.max(), arr.min())
            #     imgs_np.append(arr)

            # concat = np.concatenate(imgs_np, axis=1)
            # if TASK_ENV.take_action_cnt % 25 ==0:
            #     Image.fromarray(concat).save(f'combined_{TASK_ENV.take_action_cnt}.png')
            # ========= debug image =========

            # from IPython import embed;embed()
            env_step_before = int(TASK_ENV.take_action_cnt)
            ret = model.infer(formatted_observation) #(TASK_ENV, model, observation)
            action, latency = ret['action'], ret['server_timing']
            executed_actions = []
            if len(action.shape) == 2:
                initial_obs = False
                for act in action:
                    if initial_obs: # ensure the video is correct, but slow down simulation
                        # observation = TASK_ENV.get_obs()
                        pass
                    else:
                        initial_obs = True
                    TASK_ENV.take_action(act)
                    executed_actions.append(np.asarray(act).copy())
                    if TASK_ENV.eval_success:
                        succ = True
                        break
            else:
                TASK_ENV.take_action(action)
                executed_actions.append(np.asarray(action).copy())
                if TASK_ENV.eval_success:
                    succ = True

            if recap_recorder is not None:
                generated_action = np.asarray(action)
                if executed_actions:
                    executed_action = np.stack(executed_actions, axis=0)
                else:
                    action_dim = generated_action.shape[-1]
                    executed_action = np.empty((0, action_dim), dtype=generated_action.dtype)
                recap_recorder.record_step(
                    formatted_observation,
                    generated_action,
                    executed_action,
                    env_step_before=env_step_before,
                    env_step_after=int(TASK_ENV.take_action_cnt),
                    terminated=succ,
                    truncated=(not succ and TASK_ENV.take_action_cnt >= TASK_ENV.step_lim),
                    info={"server_latency": latency, "recap_runtime": ret.get("_recap_runtime")},
                )

            print(f"infer time {latency}")


        if recap_recorder is not None:
            recap_recorder.finalize(
                success=succ,
                terminal_reason="success" if succ else "step_limit",
                metadata={"environment_steps": int(TASK_ENV.take_action_cnt)},
            )

        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        result_tag = "success" if succ else "failure"
        
        old_name = f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4"
        new_name = f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}_{result_tag}.mp4"

        if os.path.exists(old_name):
            os.rename(old_name, new_name)
            print(f"Video saved: {new_name}")


        episode_outcomes.append(
            {"seed": int(now_seed), "success": bool(succ), "instruction": str(instruction)}
        )
        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc, episode_outcomes


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    # from test_render import Sapien_TEST
    # Sapien_TEST()

    usr_args = parse_args_and_config()

    try:
        main(usr_args)
    except CohortProtocolError:
        traceback.print_exc()
        sys.exit(PROTOCOL_ERROR_EXIT)
