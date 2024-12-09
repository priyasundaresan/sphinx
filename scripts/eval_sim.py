import argparse
import os
import time
from collections import defaultdict
import copy
import pyrallis
import numpy as np
import torch
import torch.multiprocessing as mp
from typing import Union

if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn")

import common_utils
from envs.robomimic_env import RobomimicEnv, RobomimicEnvConfig
from dataset_utils.dense_dataset import DenseInputProcessor
from models.diffusion_policy import DiffusionPolicy
from models.pointcloud_dp import DP3
from models.pointnet2_utils import farthest_point_sample


class EvalProc:
    def __init__(
        self,
        seeds,
        process_id,
        env_cfg: RobomimicEnvConfig,
        camera_names: list[str],
        image_size: int,
        terminal_queue: mp.Queue,
        record_dir=None,
    ):
        self.seeds = seeds
        self.process_id = process_id

        self.env_cfg = env_cfg
        self.camera_names = camera_names
        self.image_size = image_size
        self.record_dir = record_dir

        self.terminal_queue = terminal_queue
        self.send_queue = mp.Queue()
        self.recv_queue = mp.Queue()

    def start(self):
        env = RobomimicEnv(self.env_cfg)
        input_processor = DenseInputProcessor(self.camera_names, self.image_size)

        if self.record_dir is not None:
            recorder = common_utils.Recorder(self.record_dir)
        else:
            recorder = None

        results = {}
        for seed in self.seeds:
            np.random.seed(seed)

            env.reset()
            actions = []

            if recorder is not None:
                recorder.add_numpy(env.observe(), ["agentview_image"])

            while not env.terminal:
                # NOTE: obs["obs"] should be a cpu tensor because it
                # is more complicated to move cuda tensors around.
                if len(actions) == 0:
                    obs = env.observe()
                    obs = input_processor.process(obs)
                    self.send_queue.put((self.process_id, obs))
                    action_seq: torch.Tensor = self.recv_queue.get()
                    # print("inference:", action_seq.size())
                    actions = [action.squeeze(0).numpy() for action in action_seq.split(1, 0)]

                # print("act")
                action = actions.pop(0)
                env.apply_action(action[:3], action[3:6], action[6])

                if recorder is not None:
                    recorder.add_numpy(env.observe(), ["agentview_image"])

                if env.reward > 0:
                    # early terminate if succeed
                    break

            results[seed] = (float(env.reward), env.num_step)
            if recorder is not None:
                recorder.save(f"s{seed}")

        self.terminal_queue.put((self.process_id, results))
        return


class EvalPcdProc:
    def __init__(
        self,
        seeds,
        process_id,
        env_cfg: RobomimicEnvConfig,
        num_points: int,
        terminal_queue: mp.Queue,
        record_dir=None,
    ):
        self.seeds = seeds
        self.process_id = process_id

        self.env_cfg = env_cfg
        self.num_points = num_points
        self.record_dir = record_dir

        self.terminal_queue = terminal_queue
        self.send_queue = mp.Queue()
        self.recv_queue = mp.Queue()

    def start(self):
        env = RobomimicEnv(self.env_cfg)
        # input_processor = DenseInputProcessor(self.camera_names, self.image_size)

        if self.record_dir is not None:
            recorder = common_utils.Recorder(self.record_dir)
        else:
            recorder = None

        results = {}
        for seed in self.seeds:
            np.random.seed(seed)

            env.reset()
            actions = []

            if recorder is not None:
                recorder.add_numpy(env.observe(), ["agentview_image"])

            while not env.terminal:
                # NOTE: obs["obs"] should be a cpu tensor because it
                # is more complicated to move cuda tensors around.
                if len(actions) == 0:
                    obs = env.observe()
                    points, colors = env.get_point_cloud(obs, crop_table=True)
                    points = torch.from_numpy(points).float()
                    colors = torch.from_numpy(colors).float()

                    fps_indices = farthest_point_sample(points.unsqueeze(0), self.num_points)
                    fps_indices = fps_indices.squeeze(0)
                    points = points[fps_indices]
                    colors = colors[fps_indices]

                    points = torch.cat([points, colors], -1)
                    obs = {"points": points, "prop": torch.from_numpy(obs["proprio"]).float()}

                    self.send_queue.put((self.process_id, obs))
                    action_seq: torch.Tensor = self.recv_queue.get()
                    actions = [action.squeeze(0).numpy() for action in action_seq.split(1, 0)]

                action = actions.pop(0)
                env.apply_action(action[:3], action[3:6], action[6])

                if recorder is not None:
                    recorder.add_numpy(env.observe(), ["agentview_image"])

                if env.reward > 0:
                    # early terminate if succeed
                    break

            results[seed] = (float(env.reward), env.num_step)
            if recorder is not None:
                recorder.save(f"s{seed}")

        self.terminal_queue.put((self.process_id, results))
        return


def run_eval_seeds(
    agent: Union[DiffusionPolicy, DP3],
    env_cfg: RobomimicEnvConfig,
    seeds: list[int],
    num_proc: int,
    save_dir,
    verbose,
) -> tuple[dict, dict]:
    # env_params["device"] = "cpu"  # avoid sending cuda across processes
    env_cfg = copy.deepcopy(env_cfg)
    if env_cfg.name == "NutAssemblySquare":
        env_cfg.max_len = 300
    elif env_cfg.name == "PickPlaceCan":
        env_cfg.max_len = 200
    else:
        assert False, f"please define eval len for {env_cfg.name}"

    game_per_proc = int(np.ceil(len(seeds) / num_proc))
    print(f"running {len(seeds)} over {num_proc} processes, {game_per_proc}/proc")
    terminal_queue = mp.Queue()

    eval_procs = []
    for i in range(num_proc):
        if i * game_per_proc > len(seeds):
            break

        proc_seeds = seeds[i * game_per_proc : (i + 1) * game_per_proc]
        if isinstance(agent, DP3):
            proc = EvalPcdProc(
                proc_seeds,
                i,
                env_cfg,
                agent.cfg.num_points,
                terminal_queue,
                save_dir,
            )
        else:
            proc = EvalProc(
                proc_seeds,
                i,
                env_cfg,
                agent.camera_views,
                agent.obs_shape[-1],
                terminal_queue,
                save_dir,
            )
        eval_procs.append(proc)

    put_queues = {i: proc.recv_queue for i, proc in enumerate(eval_procs)}
    get_queues = {i: proc.send_queue for i, proc in enumerate(eval_procs)}

    processes = {i: mp.Process(target=proc.start) for i, proc in enumerate(eval_procs)}
    for _, p in processes.items():
        p.start()

    t = time.time()
    scores = {}
    episode_lens = {}
    with torch.no_grad():
        assert not agent.training
        while len(processes) > 0:
            while not terminal_queue.empty():
                term_idx, proc_results = terminal_queue.get()
                for seed, (success, episode_len) in proc_results.items():
                    scores[seed] = success
                    episode_lens[seed] = episode_len

                processes[term_idx].join()
                processes.pop(term_idx)
                get_queues.pop(term_idx)
                put_queues.pop(term_idx)

            obses = defaultdict(list)
            idxs = []
            for _, get_queue in get_queues.items():
                if get_queue.empty():
                    continue
                data = get_queue.get()
                idxs.append(data[0])
                for k, v in data[1].items():
                    obses[k].append(v)

            if len(obses) == 0:
                continue

            batch_obs = {k: torch.stack(v).cuda() for k, v in obses.items()}
            batch_action = agent.act(batch_obs)
            for idx, action in zip(idxs, batch_action):
                put_queues[idx].put(action)

    if verbose:
        print(f"total time {time.time() - t:.2f}")
        for seed in sorted(list(scores.keys())):
            print(f"seed {seed}: score: {np.mean(scores[seed]):.2f}")
        print(common_utils.wrap_ruler(""))

    return scores, episode_lens


def main():
    from scripts import train_dense

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model path")
    parser.add_argument("--num_episode", type=int, default=10)
    parser.add_argument("--num_proc", type=int, default=10)
    parser.add_argument("--seed", type=int, default=99999)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    agent, _, cfg = train_dense.load_model(args.model, "cuda", load_only_one=True)
    env_cfg_path = os.path.join(cfg.dataset.path, "env_cfg.yaml")
    env_cfg = pyrallis.load(RobomimicEnvConfig, open(env_cfg_path, "r"))  # type: ignore

    agent.train(False)
    seeds = [args.seed + i for i in range(args.num_episode)]
    scores, episode_lens = run_eval_seeds(agent, env_cfg, seeds, args.num_proc, args.save_dir, True)

    scores = list(scores.values())
    print("score:", np.mean(scores), ", len:", np.mean(list(episode_lens.values())))
    # episode_lens = list(episode_lens.values())
    # success_lens = [l for i, l in enumerate(episode_lens) if scores[i] > 0]


if __name__ == "__main__":
    torch.set_printoptions(linewidth=100)
    main()
