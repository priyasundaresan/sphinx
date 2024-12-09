from dataclasses import dataclass
import multiprocessing as mp
import json
import os
import torch
import numpy as np
import pyrallis
import cv2

import common_utils
from common_utils.eval_utils import get_reference_initial_obs, align_env_to_image, check_for_interrupt

from scripts.train_dense import load_model
from envs.franka_env import FrankaEnv, FrankaEnvConfig

from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

from interactive_scripts.dataset_recorder import DatasetRecorder, ActMode

def preprocess_image(image):
    image = cv2.resize(image, (224,224))
    image = Image.fromarray(image)
    return image

def predict_worker(obs_queue, action_queue, checkpoint, task_string):
    # Load fine-tuned OpenVLA model
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint, 
        #attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to("cuda:0")
    vla.eval()
    vla = torch.compile(vla)

    # Add dataset_stats
    with open(os.path.join(checkpoint, 'dataset_statistics.json'), 'r') as file:
        dataset_stats = json.load(file)
        task_name = list(dataset_stats.keys())[0]
        vla.norm_stats.update(dataset_stats)

    prompt = f"In: What action should the robot take to {task_string}?\nOut:" # TODO: fill in INSTRUCTION
    print('PROMPT', prompt)

    while True:
        obs = obs_queue.get()
        if obs is None:
            break  # Exit if a None value is encountered
        
        try:
            # Preprocess the image (CPU)
            raw_image = obs['agent1_image']
            image = preprocess_image(raw_image)
            inputs_cpu = processor(prompt, image)
            inputs_gpu = inputs_cpu.to("cuda:0", dtype=torch.bfloat16)

            # Predict action
            action = vla.predict_action(**inputs_gpu, unnorm_key=task_name, do_sample=False)

            # Put the action in the action queue
            action_queue.put((raw_image, action))
        except Exception as e:
            print(f"Error in worker process: {e}")


def run_episode(
    recorder: DatasetRecorder,
    processor,
    vla,
    task_name: str,
    task_string: str,
    env: FrankaEnv,
    freq: float,
    stopwatch: common_utils.Stopwatch,
    show_camera: int,
    reference_rollout_dir: str,
):
    # We can align the initial state of the environment given a reference rollout
    if reference_rollout_dir:
        episode_idx = recorder.get_next_idx()
        assert(episode_idx < len([fn for fn in os.listdir(reference_rollout_dir) if 'npz' in fn]))
        initial_image = get_reference_initial_obs(episode_idx, reference_rollout_dir)
        align_env_to_image(env, initial_image)

    with stopwatch.time("reset"):
        env.reset()

    prompt = f"In: What action should the robot take to {task_string}?\nOut:" # TODO: fill in INSTRUCTION
    print('PROMPT', prompt)

    while True:
        with common_utils.FreqGuard(freq):
            # Check if user interrupted
            if check_for_interrupt():
                print("Rollout interrupted by user.")
                break

            with stopwatch.time("observe"):
                obs = env.observe()
                raw_image = obs['agent1_image']
    
                if show_camera:
                    cv2.imshow("img", cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))
                    cv2.waitKey(1)

            image = preprocess_image(raw_image)
            inputs_cpu = processor(prompt, image)
            inputs_gpu = inputs_cpu.to("cuda:0", dtype=torch.bfloat16)

            # Predict action
            action = vla.predict_action(**inputs_gpu, unnorm_key=task_name, do_sample=False)
    
            # Apply the last received action
            ee_pos = action[:3]
            ee_euler = action[3:6]
            gripper_open = action[6]
            env.apply_action(ee_pos, ee_euler, gripper_open, is_delta=True)
            action = np.hstack((ee_pos, ee_euler, [gripper_open]))
            recorder.record(ActMode.Interpolate, obs, action)

    recorder.end_episode(save=True)
    stopwatch.summary()
    env.reset()
    return

@dataclass
class EvalConfig:
    task_string: str = ""
    weight: str = ""
    dataset: str = ""
    # original env and env overwrite
    env_cfg_path: str = ""
    # others
    show_camera: int = 1
    freq: float = 10
    reference_rollout_dir: str = "" # Optionally pass a reference directory to load initial env states from

    @property
    def env_cfg(self):
        return pyrallis.load(FrankaEnvConfig, open(self.env_cfg_path))

def main(cfg: EvalConfig):

    # assert False
    env = FrankaEnv(cfg.env_cfg)
    env.reset()

    # warm up camera for 10s
    for _ in range(int(cfg.freq) * 5):
        with common_utils.FreqGuard(cfg.freq):
            env.observe()

    stopwatch = common_utils.Stopwatch()

    recorder = DatasetRecorder('openvla_real_rollouts')

    # Load Processor & VLA from the local path
    processor = AutoProcessor.from_pretrained(cfg.weight, trust_remote_code=True)

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.weight, 
        #attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to("cuda:0")
    vla.eval()
    vla = torch.compile(vla)

    # Add dataset_stats
    with open(os.path.join(cfg.weight, 'dataset_statistics.json'), 'r') as file:
        dataset_stats = json.load(file)
        task_name = list(dataset_stats.keys())[0]
        vla.norm_stats.update(dataset_stats)

    for i in range(10):
        run_episode(recorder, processor, vla, task_name, cfg.task_string, env, cfg.freq, stopwatch, cfg.show_camera, cfg.reference_rollout_dir)

if __name__ == "__main__":
    """example command:

    python scripts/eval_openvla.py --weight /scr2/priyasun/openvla/checkpoints/openvla-7b+cup_stack+b20+lr-0.0005+lora-r32+dropout-0.0/ --env_cfg_path envs/fr3.yaml --freq 10 --task_string "put the pink cup in the yellow cup"
    """
    import rich.traceback

    rich.traceback.install()
    np.set_printoptions(precision=4, linewidth=100, suppress=True)
    torch.set_printoptions(linewidth=100, sci_mode=False)
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore

    cfg = pyrallis.parse(config_class=EvalConfig)  # type: ignore
    main(cfg)
