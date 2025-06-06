import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(RevIN, self).__init__()
        self.eps = eps
        self.num_features = num_features
        self.mean = None
        self.std = None

    def forward(self, x, mode="norm"):
        if mode == "norm":
            self.mean = x.mean(dim=(1, 2), keepdim=True)
            self.std = x.std(dim=(1, 2), keepdim=True) + self.eps
            return (x - self.mean) / self.std
        elif mode == "denorm":
            return x * self.std + self.mean
        else:
            raise ValueError("Mode must be 'norm' or 'denorm'")