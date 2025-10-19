import torch
from torch import nn


class FeatureDiversityLoss(nn.Module):
    def __init__(self, scaling_factor):
        super().__init__()
        self.scaling_factor = scaling_factor

    def forward(self, feature_maps):
        feature_maps = preserve_avg_func(feature_maps)
        flattened_feature_maps = feature_maps.flatten(2)
        batch, features, map_size = flattened_feature_maps.size()
        diversity_loss = torch.sum(torch.amax(flattened_feature_maps, dim=1))
        return -diversity_loss / batch * self.scaling_factor


def norm_vector(x):
    return x / (torch.norm(x, dim=1) + 1e-5)[:, None]


def preserve_avg_func(x):
    avgs = torch.mean(x, dim=1)
    max_avgs = torch.max(avgs, dim=0)[0]
    scaling_factor = avgs / max_avgs[..., None]
    softmaxed_maps = softmax_feature_maps(x)
    scaled_maps = softmaxed_maps * scaling_factor[..., None, None]
    return scaled_maps


def softmax_feature_maps(x):
    return torch.softmax(x.reshape(x.size(0), x.size(1), -1), 2).view_as(x)