from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn
from termcolor import cprint
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from models.cond_unet1d import ConditionalUnet1D
from models.action_normalizer import ActionNormalizer
from models.pointnet2_utils import farthest_point_sample


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: list[int],
    activation_fn=nn.ReLU,
    squash_output: bool = False,
) -> list[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud"""

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 512,
        use_layernorm: bool = True,
        final_norm: str = "layernorm",
    ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), "cyan")
        cprint("pointnet use_final_norm: {}".format(final_norm), "cyan")

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels), nn.LayerNorm(out_channels)
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.out_channels = out_channels

    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x


class DP3Encoder(nn.Module):
    def __init__(
        self,
        proprio_dim: int,
        state_mlp_size=(64, 64),
        state_mlp_activation_fn=nn.ReLU,
    ):
        super().__init__()

        self.extractor = PointNetEncoderXYZRGB()

        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]
        self.state_mlp = nn.Sequential(
            *create_mlp(proprio_dim, output_dim, net_arch, state_mlp_activation_fn)
        )
        self.out_channels = self.extractor.out_channels + output_dim
        cprint(f"[DP3Encoder] output dim: {self.out_channels}", "red")

    def forward(self, points: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        pn_feat = self.extractor(points)  # B * out_channel
        # state = observations[self.state_key]
        state_feat = self.state_mlp(proprio)  # B * 64
        final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        return final_feat


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
class DP3Config:
    # algo
    ddim: DDIMConfig = field(default_factory=lambda: DDIMConfig())
    action_horizon: int = 8
    prediction_horizon: int = 16
    shift_pad: int = 4
    # arch
    num_points: int = 1024
    base_down_dims: int = 256
    kernel_size: int = 5
    diffusion_step_embed_dim: int = 128


class DP3(nn.Module):
    def __init__(
        self,
        prop_dim: int,
        action_dim: int,
        cfg: DP3Config,
    ):
        super().__init__()
        self.prop_dim = prop_dim
        self.action_dim = action_dim
        self.cfg = cfg

        # for action normalization,
        # we use paramater here so that it will be saved with policy.state_dict()
        self.action_min = nn.Parameter(torch.zeros(action_dim) - 1, requires_grad=False)
        self.action_max = nn.Parameter(torch.zeros(action_dim) + 1, requires_grad=False)
        self.action_normalizer = ActionNormalizer(self.action_min.data, self.action_max.data)

        # nets
        self.encoder = DP3Encoder(proprio_dim=prop_dim)
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=self.encoder.out_channels,
            kernel_size=cfg.kernel_size,
            diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
            down_dims=[cfg.base_down_dims, cfg.base_down_dims * 2, cfg.base_down_dims * 4],
        )

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=cfg.ddim.num_train_timesteps,
            beta_schedule=cfg.ddim.beta_schedule,
            clip_sample=bool(cfg.ddim.clip_sample),
            set_alpha_to_one=bool(cfg.ddim.set_alpha_to_one),
            steps_offset=cfg.ddim.steps_offset,
            prediction_type=cfg.ddim.prediction_type,
        )
        self.to("cuda")

    def init_action_normalizer(self, action_min: torch.Tensor, action_max: torch.Tensor):
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
        if obs["points"].dim() == 2:
            unsqueezed = True
            for k, v in obs.items():
                obs[k] = v.unsqueeze(0)

        bsize = obs["points"].size(0)
        device = obs["points"].device

        # pure noise input to begine with
        noisy_action = torch.randn(
            (bsize, self.cfg.prediction_horizon, self.action_dim), device=device
        )

        num_inference_timesteps = self.cfg.ddim.num_inference_timesteps
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        pcd_emb = self.encoder.forward(obs["points"], obs["prop"])
        # all_actions = []
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(noisy_action, k, global_cond=pcd_emb)

            # inverse diffusion step (remove noise)
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=noisy_action  # type: ignore
            ).prev_sample.detach()  # type: ignore

        action = noisy_action[:, : self.cfg.action_horizon]
        action = self.action_normalizer.denormalize(action)

        if unsqueezed:
            action = action.squeeze(0)
        if cpu:
            action = action.cpu()
        return action

    def loss(self, batch, stopwatch, avg=True):
        with stopwatch.time("loss.fps"):
            points: torch.Tensor = batch.obs["points"]
            prop: torch.Tensor = batch.obs["prop"]

            # [num_pass, num_point]
            fps_indices = farthest_point_sample(points[:, :, :3], self.cfg.num_points)
            repeat_fps_indices = fps_indices.unsqueeze(2).repeat(1, 1, 6)
            points = points.gather(1, repeat_fps_indices)

        with stopwatch.time("loss.learn"):
            actions: torch.Tensor = batch.action["action"]
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

            pcd_emb = self.encoder.forward(points, prop)
            noise_pred = self.noise_pred_net.forward(noisy_actions, timesteps, global_cond=pcd_emb)
            # loss: [batch, num_action, action_dim]
            loss: torch.Tensor = nn.functional.mse_loss(noise_pred, noise, reduction="none").sum(2)

            assert "valid_action" in batch.obs
            valid_action: torch.Tensor = batch.obs["valid_action"]
            assert loss.size() == valid_action.size()
            loss = (loss * valid_action).sum(1) / valid_action.sum(1)

            if avg:
                loss = loss.mean()
        return loss


def test():
    enc = PointNetEncoderXYZRGB()
    print(enc)

    points = torch.rand(2, 1024, 6)
    y = enc.forward(points)
    print(">>>>>", y.size())

    proprio_dim = 10
    dp3 = DP3Encoder(proprio_dim)
    proprio = torch.rand(2, proprio_dim)
    feat = dp3.forward(points, proprio)
    print(feat.size())


if __name__ == "__main__":
    test()
