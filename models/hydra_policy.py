from dataclasses import dataclass, field
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

import common_utils
from models.dp_net import MultiviewCondUnet, MultiviewCondUnetConfig
from models.action_normalizer import ActionNormalizer
from models.diffusion_policy import DDPMConfig, DDIMConfig
import torch.nn.functional as F


@dataclass
class HydraPolicyConfig:
    # algo
    use_ddpm: int = 1
    ddpm: DDPMConfig = field(default_factory=lambda: DDPMConfig())
    ddim: DDIMConfig = field(default_factory=lambda: DDIMConfig())
    dense_action_horizon: int = 8
    dense_prediction_horizon: int = 16
    shift_pad: int = 4
    # arch
    cond_unet: MultiviewCondUnetConfig = field(default_factory=lambda: MultiviewCondUnetConfig())


class HydraPolicy(nn.Module):
    def __init__(
        self,
        obs_horizon,
        obs_shape,
        prop_dim: int,
        action_dim: int,
        camera_views,
        cfg: HydraPolicyConfig,
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.obs_shape = obs_shape
        self.prop_dim = prop_dim
        self.action_dim = action_dim
        self.camera_views = camera_views
        self.cfg = cfg

        # for data augmentation in training
        self.aug = common_utils.RandomAug(pad=cfg.shift_pad)
        self.dense_action_min = nn.Parameter(torch.zeros(action_dim) - 1, requires_grad=False)
        self.dense_action_max = nn.Parameter(torch.zeros(action_dim) + 1, requires_grad=False)
        self.dense_action_normalizer = ActionNormalizer(
            self.dense_action_min.data, self.dense_action_max.data
        )

        # we concat image in dataset & env_wrapper
        if self.obs_horizon > 1:
            obs_shape = (obs_shape[0] // self.obs_horizon, obs_shape[1], obs_shape[2])

        self.net = MultiviewCondUnet(
            obs_shape,
            obs_horizon,
            prop_dim,
            camera_views,
            action_dim,
            cfg.cond_unet,
        )

        ### waypoint and mode heads ###
        embed_dim = self.net.encoder.repr_dim
        self.waypoint_head = nn.Linear(embed_dim, action_dim)
        self.mode_head = nn.Linear(embed_dim, 3)  # waypoint, dense, or terminate

        if cfg.use_ddpm:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=cfg.ddpm.num_train_timesteps,
                beta_schedule=cfg.ddpm.beta_schedule,
                clip_sample=bool(cfg.ddpm.clip_sample),
                prediction_type=cfg.ddpm.prediction_type,
            )
        else:
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=cfg.ddim.num_train_timesteps,
                beta_schedule=cfg.ddim.beta_schedule,
                clip_sample=bool(cfg.ddim.clip_sample),
                set_alpha_to_one=bool(cfg.ddim.set_alpha_to_one),
                steps_offset=cfg.ddim.steps_offset,
                prediction_type=cfg.ddim.prediction_type,
            )
        self.to("cuda")

    def init_dense_action_normalizer(
        self, dense_action_min: torch.Tensor, dense_action_max: torch.Tensor
    ):
        # for dense_action normalization,
        # we use paramater here so that it will be saved with policy.state_dict()
        self.dense_action_min.data.copy_(dense_action_min)
        self.dense_action_max.data.copy_(dense_action_max)
        self.dense_action_normalizer = ActionNormalizer(
            self.dense_action_min.data, self.dense_action_max.data
        )
        self.dense_action_normalizer.to(self.dense_action_min.device)
        print("creating dense_action normalizer with")
        print("  scale:", self.dense_action_normalizer.scale.squeeze())
        print("  offset:", self.dense_action_normalizer.offset.squeeze())

    def to(self, device):
        self.dense_action_normalizer.to(device)
        return super().to(device)

    def load_state_dict(self, state_dict, strict: bool = True):
        super().load_state_dict(state_dict, strict=strict)
        self.dense_action_normalizer = ActionNormalizer(
            self.dense_action_min.data, self.dense_action_max.data
        )
        print("creating dense_action normalizer with")
        print("  scale:", self.dense_action_normalizer.scale.squeeze())
        print("  offset:", self.dense_action_normalizer.offset.squeeze())

    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], *, cpu=True):
        assert not self.training

        unsqueezed = False
        if obs[self.camera_views[0]].dim() == 3:
            unsqueezed = True
            for k, v in obs.items():
                obs[k] = v.unsqueeze(0)

        bsize = obs[self.camera_views[0]].size(0)
        device = obs[self.camera_views[0]].device

        # pure noise input to begine with
        noisy_dense_action = torch.randn(
            (bsize, self.cfg.dense_prediction_horizon, self.action_dim), device=device
        )

        if self.cfg.use_ddpm:
            num_inference_timesteps = self.cfg.ddpm.num_inference_timesteps
        else:
            num_inference_timesteps = self.cfg.ddim.num_inference_timesteps
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        cached_image_emb = None
        all_dense_actions = []
        for k in self.noise_scheduler.timesteps:
            noise_pred, cached_image_emb = self.net.predict_noise(
                obs, noisy_dense_action, k, cached_image_emb
            )

            # inverse diffusion step (remove noise)
            noisy_dense_action = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=noisy_dense_action  # type: ignore
            ).prev_sample.detach()  # type: ignore

            all_dense_actions.append(noisy_dense_action)

        dense_action = noisy_dense_action
        # when obs_horizon=2, the model was trained as
        # o_0, o_1,
        # a_0, a_1, a_2, a_3, ..., a_{h-1}  -> dense_action_horizon number of predictions
        # so we DO NOT use the first prediction at test time
        dense_action = dense_action[:, self.obs_horizon - 1 : self.cfg.dense_action_horizon]

        dense_action = self.dense_action_normalizer.denormalize(dense_action)

        waypoint_action = self.waypoint_head(cached_image_emb)
        waypoint_action[:, 6] = nn.functional.sigmoid(waypoint_action[:, 6])

        target_mode_logits = self.mode_head(cached_image_emb)
        target_mode_probs = nn.functional.softmax(target_mode_logits)
        #target_mode = target_mode_probs.argmax()

        if unsqueezed:
            dense_action = dense_action.squeeze(0)
            waypoint_action = waypoint_action.squeeze(0)
            target_mode_probs = target_mode_probs.squeeze(0)
        if cpu:
            dense_action = dense_action.cpu()
            waypoint_action = waypoint_action.cpu()
            target_mode_probs = target_mode_probs.cpu()

        return dense_action, waypoint_action, target_mode_probs

    def loss(self, batch, avg=True, aug=True):
        obs = {}
        for k, v in batch.obs.items():
            if aug and (k in self.camera_views):
                obs[k] = self.aug(v.float())
            else:
                obs[k] = v.float()

        # Separate batch into dense action, waypoint action, and target mode
        dense_actions = batch.action["dense_action"]
        waypoint_actions = batch.action["waypoint_action"]
        target_modes = batch.action["target_mode"]

        ### dense pred ###
        dense_actions = self.dense_action_normalizer.normalize(dense_actions)
        assert dense_actions.min() >= -1.001 and dense_actions.max() <= 1.001

        bsize = dense_actions.size(0)
        noise = torch.randn(dense_actions.shape, device=dense_actions.device)

        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config["num_train_timesteps"],
            size=(bsize,),
            device=dense_actions.device,
        ).long()
        noisy_dense_actions = self.noise_scheduler.add_noise(dense_actions, noise, timesteps)  # type: ignore

        noise_pred, obs_emb = self.net.predict_noise(obs, noisy_dense_actions, timesteps)
        dense_loss = nn.functional.mse_loss(noise_pred, noise, reduction="none").sum(2)

        assert "valid_dense_action" in batch.obs
        valid_dense_action = batch.obs["valid_dense_action"]
        assert dense_loss.size() == valid_dense_action.size()
        dense_loss = (dense_loss * valid_dense_action).sum(1) / valid_dense_action.sum(1)
        ###

        ### waypoint pred ###
        waypoint_actions_logits = self.waypoint_head(obs_emb)
        gripper_actions_logits = waypoint_actions_logits[:, 6]
        waypoint_pose_loss = nn.functional.mse_loss(
            waypoint_actions_logits[:, :6], waypoint_actions[:, :6]
        )
        waypoint_gripper_loss = F.binary_cross_entropy_with_logits(
            gripper_actions_logits, waypoint_actions[:, 6]
        )
        waypoint_gripper_preds = (nn.functional.sigmoid(gripper_actions_logits) > 0.5).float()
        waypoint_gripper_acc = (
            (waypoint_gripper_preds == waypoint_actions[:, 6]).float().mean().item()
        )
        ###

        ### mode pred ###
        target_modes_logits = self.mode_head(obs_emb)
        mode_loss = F.cross_entropy(target_modes_logits, target_modes)
        target_modes_pred = nn.functional.softmax(target_modes_logits).argmax(dim=1)
        mode_acc = (target_modes_pred == target_modes).float().mean().item()
        ###

        if avg:
            dense_loss = dense_loss.mean()
            waypoint_pose_loss = waypoint_pose_loss.mean()
            waypoint_gripper_loss = waypoint_gripper_loss.mean()
            mode_loss = mode_loss.mean()

        return (
            dense_loss,
            waypoint_pose_loss,
            waypoint_gripper_loss,
            waypoint_gripper_acc,
            mode_loss,
            mode_acc,
        )


def test():
    from dataset_utils.hydra_dataset import HydraDataset, HydraDatasetConfig

    dataset_cfg = HydraDatasetConfig(
        path="data/cups2",
        camera_views="agent1_image",
        image_size=96,
    )
    dataset = HydraDataset(dataset_cfg, load_only_one=True)

    obs_shape = dataset.obs_shape
    prop_dim = dataset.prop_dim
    action_dim = dataset.action_dim
    batch_size = 2
    policy_cfg = HydraPolicyConfig()
    policy = HydraPolicy(
        1, obs_shape, prop_dim, action_dim, dataset.camera_views, policy_cfg
    ).cuda()

    batch = dataset.sample_dp(batch_size, policy_cfg.dense_prediction_horizon, "cuda:0")
    loss = policy.loss(batch)

    obs = dataset.episodes[0][0]
    policy.train(False)

    obs = {k: obs[k].cuda() for k in obs}
    dense_action, waypoint_action, target_mode = policy.act(obs)
    print(dense_action, waypoint_action, target_mode)


if __name__ == "__main__":
    test()
