import math
from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
import numpy as np
import einops
from scipy.spatial.transform import Rotation as R

from models.pointnet2_utils import farthest_point_sample


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_head):
        super().__init__()
        assert embed_dim % num_head == 0

        self.num_head = num_head
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_mask: Optional[torch.Tensor]):
        """
        x: [batch, seq, embed_dim]
        """
        qkv = self.qkv_proj(x)
        q, k, v = einops.rearrange(qkv, "b t (k h d) -> b k h t d", k=3, h=self.num_head).unbind(1)
        # force flash/mem-eff attention, it will raise error if flash/mem-eff cannot be applied
        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool

        with torch.backends.cuda.sdp_kernel(enable_math=False):
            attn_v = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, attn_mask=attn_mask
            )
        attn_v = einops.rearrange(attn_v, "b h t d -> b t (h d)")
        return self.out_proj(attn_v)


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_head, dropout):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_head)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear2 = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def mlp(self, x):
        x = self.linear2(nn.functional.gelu(self.linear1(x)))
        return x

    def forward(self, x, attn_mask=None):
        x = x + self.dropout(self.mha(self.ln1(x), attn_mask))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


@dataclass
class WaypointTransformerConfig:
    # sizes
    preset: str = ""
    num_layer: int = 6
    embed_dim: int = 512
    num_head: int = 8
    # other design choice
    drop: float = 0
    custom_init: int = 0
    final_ln: int = 1
    topk_train: int = 50
    topk_eval: int = 3
    # model cfg
    use_euler: int = 1  # euler or quat
    npoints: int = 1024
    pred_off: int = 1
    pred_point: int = 1
    per_point_rot: int = 0

    def __post_init__(self):
        if self.preset == "small":
            self.num_layer = 6
            self.embed_dim = 512
            self.num_head = 8
        elif self.preset == "medium":
            self.num_layer = 12
            self.embed_dim = 768
            self.num_head = 12
        else:
            assert self.preset == ""


class WaypointTransformer(nn.Module):
    def __init__(self, cfg: WaypointTransformerConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.pred_off:
            assert cfg.pred_point
        if cfg.per_point_rot:
            assert cfg.pred_off

        self.input_embed = nn.Linear(6, cfg.embed_dim)
        self.pos_rot_gripper_embed = nn.Parameter(torch.zeros(1, 3, cfg.embed_dim))
        self.proprio_embed = nn.Linear(7, cfg.embed_dim)
        torch.nn.init.normal_(self.pos_rot_gripper_embed, mean=0.0, std=0.02)
        self.layers = nn.ModuleList(
            [TransformerLayer(cfg.embed_dim, cfg.num_head, cfg.drop) for _ in range(cfg.num_layer)]
        )
        self.final_ln = nn.LayerNorm(cfg.embed_dim)
        if cfg.per_point_rot:
            self.points_output = nn.Linear(cfg.embed_dim, 4 + 3)  # click + offset
        else:
            self.points_output = nn.Linear(cfg.embed_dim, 4)  # click + offset
        self.pos_output = nn.Linear(cfg.embed_dim, 3)
        self.rot_output = nn.Linear(cfg.embed_dim, 3 if self.cfg.use_euler else 4)
        self.gripper_output = nn.Linear(cfg.embed_dim, 1)
        self.mode_output = nn.Linear(cfg.embed_dim, 3)

        # this custom init does not work well
        if cfg.custom_init:
            # init all weights
            self.apply(self._init_weights)
            # apply special scaled init to the residual projections, per GPT-2 paper
            for pn, p in self.named_parameters():
                if pn.endswith("linear2.weight") or pn.endswith("linear1.weight"):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.num_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, points: torch.Tensor, proprio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        return:
            click_logits: [batch, num_point]
            rot: [batch, rot_dim (3 or 4)]
            gripper_logit: [batch]
        """
        # points: [batch, num_point, 6]
        bsize, num_point, _ = points.size()
        assert num_point == self.cfg.npoints

        point_embeds = self.input_embed(points)
        proprio_embed = self.proprio_embed(proprio)
        pos_rot_gripper_embed = self.pos_rot_gripper_embed.repeat(bsize, 1, 1)
        x: torch.Tensor = torch.cat(
            [point_embeds, pos_rot_gripper_embed, proprio_embed.unsqueeze(1)], dim=1
        )

        for layer in self.layers:
            x = layer(x)
        x = self.final_ln(x)

        points_feat, pos_feat, rot_feat, gripper_feat, mode_feat = x.split(
            [num_point, 1, 1, 1, 1], dim=1
        )

        # per-point predictions: click, pos
        points_out = self.points_output.forward(points_feat)
        if self.cfg.per_point_rot:
            click_logits, points_off, rot = points_out.split([1, 3, 3], dim=2)
        else:
            click_logits, points_off = points_out.split([1, 3], dim=2)
            rot = self.rot_output.forward(rot_feat.squeeze(1))

        click_logits = click_logits.squeeze(-1)
        # global predictions: rot, gripper, mode (TODO)
        pos = self.pos_output.forward(pos_feat.squeeze(1))
        gripper_logit = self.gripper_output.forward(gripper_feat.squeeze(1)).squeeze(1)
        mode_logit = self.mode_output.forward(mode_feat.squeeze(1))
        return click_logits, points_off, pos, rot, gripper_logit, mode_logit

    @torch.no_grad
    def inference(
        self, points: torch.Tensor, colors: torch.Tensor, proprio: torch.Tensor, num_pass: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.int64]:
        """
        args:
            points: [N, 3]
            colors: [N, 3]
            proprio: [proprio_dim]
        """
        assert not self.training
        # print(f"{points.size()}")
        points = points.unsqueeze(0).repeat(num_pass, 1, 1).cuda()
        colors = colors.unsqueeze(0).repeat(num_pass, 1, 1).cuda()
        fps_indices = farthest_point_sample(points, self.cfg.npoints)  # [num_pass, num_point]
        repeat_fps_indices = fps_indices.unsqueeze(2).repeat(1, 1, 3)
        points = points.gather(1, repeat_fps_indices)
        colors = colors.gather(1, repeat_fps_indices)

        # points/colors: [num_pass, num_point, 6]
        points = torch.cat([points, colors], dim=2)
        proprio = proprio.unsqueeze(0).repeat(num_pass, 1).cuda()
        click_logits, points_off, pos, rot, gripper_logit, mode_logit = self.forward(
            points, proprio
        )

        _, sorted_click_indices = click_logits.sort(dim=1, descending=True)
        # click_indices: [num_pass, topk_eval]
        click_indices = sorted_click_indices[:, : self.cfg.topk_eval]
        unsampled_click_indices = fps_indices.gather(1, click_indices).flatten().cpu().numpy()

        # compute action pos
        if self.cfg.pred_off:
            selected_points = points.gather(1, click_indices.unsqueeze(2).repeat(1, 1, 6))
            selected_xyz = selected_points[:, :, :3]
            selected_off = points_off.gather(1, click_indices.unsqueeze(2).repeat(1, 1, 3))

            assert selected_xyz.size() == selected_off.size()
            target_pos = (selected_xyz - selected_off).flatten(0, 1).mean(0).cpu().numpy()
        else:
            target_pos = pos.mean(0).cpu().numpy()

        # average rot
        if self.cfg.per_point_rot:
            selected_rot = rot.gather(1, click_indices.unsqueeze(2).repeat(1, 1, 3))
            rot = selected_rot.flatten(0, 1)

        assert self.cfg.use_euler
        rot_quat = R.from_euler("xyz", rot.cpu().numpy()).as_quat()  # type: ignore
        rot_quat = np.mean(rot_quat, axis=0)
        rot_quat /= np.linalg.norm(rot_quat)
        target_rot = R.from_quat(rot_quat).as_euler("xyz")

        gripper = nn.functional.sigmoid(gripper_logit).mean()
        gripper = gripper.round().item()

        mode_probs = nn.functional.softmax(mode_logit, dim=-1)
        mode_probs = mode_probs.squeeze().detach().cpu().numpy()
        mode = mode_probs.argmax()

        return unsampled_click_indices, target_pos, target_rot, gripper, mode

    @torch.no_grad
    def inference_click_probs(
        self, points: torch.Tensor, colors: torch.Tensor, proprio: torch.Tensor
    ) -> torch.Tensor:
        assert not self.training
        unsampled_click_probs = torch.zeros(points.size(0)).cuda()

        points = points.unsqueeze(0).repeat(1, 1, 1).cuda()
        colors = colors.unsqueeze(0).repeat(1, 1, 1).cuda()
        fps_indices = farthest_point_sample(points, self.cfg.npoints)  # [num_pass, num_point]
        repeat_fps_indices = fps_indices.unsqueeze(2).repeat(1, 1, 3)
        points = points.gather(1, repeat_fps_indices)
        colors = colors.gather(1, repeat_fps_indices)

        # points/colors: [num_pass, num_point, 6]
        points = torch.cat([points, colors], dim=2)
        proprio = proprio.unsqueeze(0).repeat(1, 1).cuda()
        click_logits, *_ = self.forward(points, proprio)
        click_probs = nn.functional.softmax(click_logits, -1).squeeze(0)

        unsampled_click_probs.scatter_(0, fps_indices.squeeze(0), click_probs)
        return unsampled_click_probs.cpu()
