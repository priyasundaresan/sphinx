from dataclasses import dataclass, field
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

import common_utils
from models.dp_net import MultiviewCondUnet, MultiviewCondUnetConfig
from models.action_normalizer import ActionNormalizer

@dataclass
class DDPMConfig:
    num_train_timesteps: int = 100
    num_inference_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: int = 1
    prediction_type: str = "epsilon"


@dataclass
class DDIMConfig:
    num_train_timesteps: int = 100
    num_inference_timesteps: int = 10
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: int = 1
    set_alpha_to_one: int = 1
    steps_offset: int = 0
    prediction_type: str = "epsilon"

@dataclass
class DiffusionPolicyConfig:
    # algo
    use_ddpm: int = 1
    ddpm: DDPMConfig = field(default_factory=lambda: DDPMConfig())
    ddim: DDIMConfig = field(default_factory=lambda: DDIMConfig())
    action_horizon: int = 8
    prediction_horizon: int = 16
    shift_pad: int = 4
    # arch
    cond_unet: MultiviewCondUnetConfig = field(default_factory=lambda: MultiviewCondUnetConfig())


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        obs_horizon,
        obs_shape,
        prop_dim: int,
        action_dim: int,
        camera_views,
        cfg: DiffusionPolicyConfig,
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
        # for action normalization,
        # we use paramater here so that it will be saved with policy.state_dict()
        self.action_min = nn.Parameter(torch.zeros(action_dim) - 1, requires_grad=False)
        self.action_max = nn.Parameter(torch.zeros(action_dim) + 1, requires_grad=False)
        self.action_normalizer = ActionNormalizer(self.action_min.data, self.action_max.data)

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

    def init_action_normalizer(
        self, action_min: torch.Tensor, action_max: torch.Tensor
    ):
        # for action normalization,
        # we use paramater here so that it will be saved with policy.state_dict()
        self.action_min.data.copy_(action_min)
        self.action_max.data.copy_(action_max)
        self.action_normalizer = ActionNormalizer(self.action_min.data, self.action_max.data)
        self.action_normalizer.to(self.action_min.device)
        print("creating action normalizer with")
        print("  scale:", self.action_normalizer.scale.squeeze())
        print("  offset:", self.action_normalizer.offset.squeeze())

    def to(self, device):
        self.action_normalizer.to(device)
        return super().to(device)

    def load_state_dict(self, state_dict, strict: bool = True):
        super().load_state_dict(state_dict, strict=strict)
        self.action_normalizer = ActionNormalizer(self.action_min.data, self.action_max.data)
        print("creating action normalizer with")
        print("  scale:", self.action_normalizer.scale.squeeze())
        print("  offset:", self.action_normalizer.offset.squeeze())

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
        noisy_action = torch.randn(
            (bsize, self.cfg.prediction_horizon, self.action_dim), device=device
        )

        if self.cfg.use_ddpm:
            num_inference_timesteps = self.cfg.ddpm.num_inference_timesteps
        else:
            num_inference_timesteps = self.cfg.ddim.num_inference_timesteps
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        cached_image_emb = None
        all_actions = []
        for k in self.noise_scheduler.timesteps:
            noise_pred, cached_image_emb = self.net.predict_noise(
                obs, noisy_action, k, cached_image_emb
            )

            # inverse diffusion step (remove noise)
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=noisy_action  # type: ignore
            ).prev_sample.detach()  # type: ignore

            all_actions.append(noisy_action)

        action = noisy_action
        # when obs_horizon=2, the model was trained as
        # o_0, o_1,
        # a_0, a_1, a_2, a_3, ..., a_{h-1}  -> action_horizon number of predictions
        # so we DO NOT use the first prediction at test time
        action = action[:, self.obs_horizon - 1 : self.cfg.action_horizon]

        action = self.action_normalizer.denormalize(action)

        if unsqueezed:
            action = action.squeeze(0)
        if cpu:
            action = action.cpu()
        return action

    def loss(self, batch, avg=True, aug=True):
        obs = {}
        for k, v in batch.obs.items():
            if aug and (k in self.camera_views):
                obs[k] = self.aug(v.float())
            else:
                obs[k] = v.float()

        actions = batch.action["action"]
        actions = self.action_normalizer.normalize(actions)
        # print(actions.min(), actions.max())
        assert actions.min() >= -1.001 and actions.max() <= 1.001

        bsize = actions.size(0)
        noise = torch.randn(actions.shape, device=actions.device)
        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config["num_train_timesteps"],
            size=(bsize,),
            device=actions.device,
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)  # type: ignore

        noise_pred = self.net.predict_noise(obs, noisy_actions, timesteps)[0]
        # loss: [batch, num_action, action_dim]
        loss = nn.functional.mse_loss(noise_pred, noise, reduction="none").sum(2)

        assert "valid_action" in batch.obs
        valid_action = batch.obs["valid_action"]
        assert loss.size() == valid_action.size()
        loss = ((loss * valid_action).sum(1) / valid_action.sum(1))

        if avg:
            loss = loss.mean()
        return loss


def test():
    obs_shape = (3, 96, 96)
    prop_dim = 9
    action_dim = 7
    camera_views = ["agentview"]
    cfg = DiffusionPolicyConfig()

    policy = DiffusionPolicy(1, obs_shape, prop_dim, action_dim, camera_views, cfg).cuda()
    policy.train(False)

    obs = {
        "agentview": torch.rand(1, 3, 96, 96).cuda(),
        "prop": torch.rand(1, 9).cuda(),
    }
    policy.act(obs)


if __name__ == "__main__":
    test()
