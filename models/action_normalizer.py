import torch


class ActionNormalizer:
    def __init__(self, action_min: torch.Tensor, action_max: torch.Tensor):
        assert action_min.dim() == 1 and action_min.size() == action_max.size()
        assert action_min.dtype == torch.float32
        assert action_max.dtype == torch.float32

        self.action_dim = action_min.size(0)
        self.min = action_min
        self.max = action_max

        self.scale = 2 / (self.max - self.min)
        self.offset = (self.max + self.min) / (self.min - self.max)

        self.scale = self.scale.unsqueeze(0)
        self.offset = self.offset.unsqueeze(0)

    def to(self, device):
        self.scale = self.scale.to(device)
        self.offset = self.offset.to(device)

    def normalize(self, value: torch.Tensor):
        shape = value.size()
        value = value.reshape(-1, self.action_dim)
        normed_value = value * self.scale + self.offset
        normed_value = normed_value.reshape(shape)
        return normed_value

    def denormalize(self, normed_value: torch.Tensor):
        shape = normed_value.size()
        normed_value = normed_value.reshape(-1, self.action_dim)
        value = (normed_value - self.offset) / self.scale
        value = value.reshape(shape)
        return value
