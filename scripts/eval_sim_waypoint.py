import argparse
import copy
import numpy as np
import torch
import os
import pyrallis

from envs.robomimic_env import RobomimicEnv, RobomimicEnvConfig
from models.waypoint_transformer import WaypointTransformer
import common_utils


@torch.no_grad
def eval_waypoint(
    policy: WaypointTransformer,
    env_cfg: RobomimicEnvConfig,
    seed: int,
    num_pass: int,
    save_dir,
    record: bool,
):
    assert not policy.training

    env_cfg = copy.deepcopy(env_cfg)
    if env_cfg.name == "NutAssemblySquare":
        env_cfg.max_len = 300
    elif env_cfg.name == "PickPlaceCan":
        env_cfg.max_len = 200
    else:
        assert False, "please define eval len"

    env = RobomimicEnv(env_cfg, verbose=False, on_screen_render=True)
    np.random.seed(seed)
    env.reset(render=True)

    recorder = None
    if record:
        assert save_dir is not None
        recorder = common_utils.Recorder(save_dir)

    freeze_counter = 0
    while not env.terminal:
        obs = env.observe()
        if recorder is not None:
            recorder.add_numpy(obs, ["agentview_image"])

        points, colors = env.get_point_cloud(obs, crop_table=True)
        proprio = obs["proprio"]

        _, pos_cmd, euler_cmd, gripper_cmd, _ = policy.inference(
            torch.from_numpy(points).float(),
            torch.from_numpy(colors).float(),
            torch.from_numpy(proprio).float(),
            num_pass=num_pass,
        )
        prev_num_step = env.num_step
        env.move_to(pos_cmd, euler_cmd, float(gripper_cmd), render=True)
        if env.num_step == prev_num_step:
            freeze_counter += 1
        else:
            freeze_counter = 0

        if env.reward > 0 or freeze_counter >= 3:
            break

    if recorder is not None:
        recorder.save_images(f"s{seed}")

    return env.reward, env.num_step


def _eval_waypoint_multi_episode(
    policy: WaypointTransformer,
    num_pass: int,
    env_cfg: RobomimicEnvConfig,
    seed: int,
    num_episode: int,
):
    scores = []
    num_steps = []
    for seed in range(seed, seed + num_episode):
        score, num_step = eval_waypoint(
            policy, env_cfg, seed=seed, num_pass=num_pass, save_dir=None, record=False
        )
        scores.append(score)
        num_steps.append(num_step)
    return np.mean(scores), np.mean(num_steps)


def eval_waypoint_policy(
    policy: WaypointTransformer,
    env_cfg_path: str,
    num_pass: int,
    num_eval_episode: int,
    stat: common_utils.MultiCounter,
    prefix: str = "",
):
    assert os.path.exists(env_cfg_path), f"cannot locate env config {env_cfg_path}"
    env_cfg = pyrallis.load(RobomimicEnvConfig, open(env_cfg_path, "r"))  # type: ignore
    score, num_step = _eval_waypoint_multi_episode(
        policy, num_pass, env_cfg, 99999, num_eval_episode
    )
    stat[f"eval/{prefix}score"].append(score)
    stat[f"eval/{prefix}num_step"].append(num_step)
    return score


def main():
    import os
    import sys
    from scripts.train_waypoint import load_waypoint

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model path")
    parser.add_argument("--seed", type=int, default=99999)
    parser.add_argument("--num_episode", type=int, default=10)
    parser.add_argument("--topk", type=int, default=-1)
    parser.add_argument("--num_pass", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default="eval_sim_record")
    parser.add_argument("--record", type=int, default=1)
    args = parser.parse_args()

    if args.save_dir is not None:
        log_path = os.path.join(args.save_dir, "eval.log")
        sys.stdout = common_utils.Logger(log_path, print_to_stdout=True)

    model_zoo = {
        "layer6_off": [
            "exps/waypoint/wp_all2/num_layer6_seed1/latest.pt",
            "exps/waypoint/wp_all2/num_layer6_seed2/latest.pt"
        ],
        "layer6_drop": [
            "exps/waypoint/wp_all2/use_dist1_topk_train0_drop0.1_epoch100_seed1/latest.pt",
            "exps/waypoint/wp_all2/use_dist1_topk_train0_drop0.1_epoch100_seed2/latest.pt"
        ],
        "layer6_abs": [
            "exps/waypoint/wp_all2/num_layer6_pred_off0_seed1/latest.pt",
            "exps/waypoint/wp_all2/num_layer6_pred_off0_seed2/latest.pt"
        ],
        "layer12_off": [
            "exps/waypoint/wp_all2/num_layer6_pred_off0_seed1/latest.pt",
            "exps/waypoint/wp_all2/num_layer6_pred_off0_seed2/latest.pt"
        ],
        "layer12_aug": [
            "exps/waypoint/wp_large/aug_interpolate0.2_repeat20_use_ema1_seed1/latest.pt",
            "exps/waypoint/wp_large/aug_interpolate0.2_repeat20_use_ema1_seed2/latest.pt",
        ],
        "layer12_aug_ema": [
            "exps/waypoint/wp_large/aug_interpolate0.2_repeat20_use_ema1_seed1/ema.pt",
            "exps/waypoint/wp_large/aug_interpolate0.2_repeat20_use_ema1_seed2/ema.pt",
        ],
    }
    if args.model in model_zoo:
        models = model_zoo[args.model]
    else:
        models = [args.model]

    model_scores = []
    for model in models:
        print(f">>>>>>>>>>{model}<<<<<<<<<<")
        policy, env_cfg = load_waypoint(model)
        assert isinstance(env_cfg, RobomimicEnvConfig)
        policy.train(False)
        policy = policy.cuda()

        if args.topk > 0:
            print(f"Overriding topk_eval to be {args.topk}")
            policy.cfg.topk_eval = args.topk
        else:
            print(f"Eval with original topk_eval {policy.cfg.topk_eval}")

        scores = []
        for seed in range(args.seed, args.seed + args.num_episode):
            score, _ = eval_waypoint(
                policy,
                env_cfg,
                seed=seed,
                num_pass=args.num_pass,
                save_dir=args.save_dir,
                record=args.record,
            )
            scores.append(score)

        print(f"{model}")
        print(f"score: {np.mean(scores):.4f} topk: {policy.cfg.topk_eval}, npass: {args.num_pass}")
        model_scores.append(np.mean(scores))
        print(common_utils.wrap_ruler("", max_len=80))

    print(f"average score {np.mean(model_scores):.4f}")


if __name__ == "__main__":
    main()
