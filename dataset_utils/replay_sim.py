import argparse
import os
import pyrallis
import numpy as np
import matplotlib.pyplot as plt

import common_utils
from envs.robomimic_env import RobomimicEnv, RobomimicEnvConfig
from interactive_scripts.dataset_recorder import ActMode, DatasetRecorder


def replay_demo(demo_path, verbose, render):
    demo = np.load(demo_path, allow_pickle=True)["arr_0"]

    env_cfg_path = os.path.join(os.path.dirname(demo_path), "env_cfg.yaml")
    env_cfg = pyrallis.load(RobomimicEnvConfig, open(env_cfg_path, "r"))  # type: ignore
    env = RobomimicEnv(env_cfg, render, verbose)

    env.reset_to_sim_state(demo[0]["obs"]["sim_state"])
    assert (env.observe_proprio().eef_pos - demo[0]["obs"]["ee_pos"]).max() <= 1e-7
    assert (env.observe_proprio().eef_euler - demo[0]["obs"]["ee_euler"]).max() <= 1e-7

    # next, apply action & get reward
    demo = [step for step in demo if step["mode"] in [ActMode.Dense, ActMode.Interpolate]]
    stopwatch = common_utils.Stopwatch()

    actions = []

    for i in range(len(demo)):
        with stopwatch.time("act"):
            action = demo[i]["action"]
            actions.append(action)
            env.apply_action(action[:3], action[3:6], action[-1])

        if verbose:
            with stopwatch.time("check"):
                if (i + 1) < len(demo):
                    prop = env.observe_proprio()
                    pos_diff = (prop.eef_pos - demo[i + 1]["obs"]["ee_pos"]).max()
                    gripper_diff = (prop.gripper_open - demo[i + 1]["obs"]["gripper_open"]).max()
                    print(i, f"pos: {pos_diff:.4f}, gripper: {gripper_diff:.4f}")

        if render:
            with stopwatch.time("render"):
                env.env.render()

        if demo[i]["reward"] != env.reward:
            print(
                f"Reward mismatch at {i}/{len(demo)}, "
                f"recorded {demo[i]['reward']}, real {env.reward}"
            )

    print(f"env.reward: {env.reward}")
    if render:
        fig, ax = common_utils.generate_grid(3, 1, figsize=7)
        actions = np.stack(actions)
        ax[0].plot(actions[:, 0], label="x")
        ax[0].plot(actions[:, 1], label="y")
        ax[0].plot(actions[:, 2], label="z")
        ax[0].legend()
        ax[1].plot(actions[:, 3], label="rot-x")
        ax[1].plot(actions[:, 4], label="rot-y")
        ax[1].plot(actions[:, 5], label="rot-z")
        ax[1].legend()
        ax[2].plot(actions[:, 6], label="gripper")
        ax[2].legend()
        fig.tight_layout()
        plt.show()
    return env.reward


def replay_demo_waypoint(demo_path, render, recorder):
    demo = np.load(demo_path, allow_pickle=True)["arr_0"]

    env_cfg_path = os.path.join(os.path.dirname(demo_path), "env_cfg.yaml")
    env_cfg = pyrallis.load(RobomimicEnvConfig, open(env_cfg_path, "r"))  # type: ignore
    env = RobomimicEnv(env_cfg, render)

    env.reset_to_sim_state(demo[0]["obs"]["sim_state"])
    assert (env.observe_proprio().eef_pos - demo[0]["obs"]["ee_pos"]).max() <= 1e-7
    assert (env.observe_proprio().eef_euler - demo[0]["obs"]["ee_euler"]).max() <= 1e-7

    # next, apply action & get reward
    demo = [step for step in demo if step["mode"] in [ActMode.Waypoint]]
    stopwatch = common_utils.Stopwatch()

    actions = []

    for i in range(len(demo)):
        with stopwatch.time("act"):
            action = demo[i]["action"]
            actions.append(action)
            if recorder is not None:
                recorder.record(ActMode.Waypoint, env.observe(), action, click_pos=demo[i]["click"])

            env.move_to(action[:3], action[3:6], action[-1], render=render, recorder=recorder)

    if recorder is not None:
        recorder.end_episode(save=True)
    print(f"env.reward: {env.reward}")
    return env.reward


def replay_folder(folder, waypoint, recorder):
    npz_files = list(sorted(common_utils.get_all_files(folder, "npz")))

    scores = []
    for demo in npz_files:
        print(f"replaying {demo}")
        if waypoint:
            score = replay_demo_waypoint(demo, False, recorder)
        else:
            score =  replay_demo(demo, False, False)
        print("=" * 80)
        scores.append(score)
    print(f"success rate {np.mean(scores)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=str, default=None)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--waypoint", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--render", type=int, default=0)
    parser.add_argument("--regen_folder", type=str, default=None)
    args = parser.parse_args()

    np.set_printoptions(precision=4, linewidth=100, suppress=True)

    if args.regen_folder is not None:
        recorder = DatasetRecorder(args.regen_folder)
    else:
        recorder = None

    if args.demo:
        if args.waypoint:
            replay_demo_waypoint(args.demo, args.render, recorder)
        else:
            replay_demo(args.demo, args.verbose, args.render)
    elif args.folder:
        replay_folder(args.folder, args.waypoint, recorder)
