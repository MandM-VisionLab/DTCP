import torch
from torch import nn


class FeatureDiversityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature_maps):
        assert feature_maps.dim() == 4, (
            f"FDL expects 4-D (B,C,H,W); got {tuple(feature_maps.shape)}."
        )
        feature_maps = preserve_avg_func(feature_maps)
        flat = feature_maps.flatten(2)
        batch = flat.size(0)
        diversity_loss = torch.sum(torch.amax(flat, dim=1))
        return -diversity_loss / batch


def preserve_avg_func(x):
    avgs = torch.mean(x, dim=[2, 3])
    max_avgs = torch.max(avgs, dim=1)[0]
    scaling_factor = avgs / (max_avgs[..., None] + 1e-8)
    softmaxed = softmax_feature_maps(x)
    return softmaxed * scaling_factor[..., None, None]


def softmax_feature_maps(x):
    return torch.softmax(x.reshape(x.size(0), x.size(1), -1), 2).view_as(x)