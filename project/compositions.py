import torch
import torch.nn as nn


class Add(nn.Module):
    def forward(self, x, y):
        return x + y


class Mean(nn.Module):
    def forward(self, x, y):
        mean = torch.mean(torch.stack((x, y)), dim=0)
        return mean
