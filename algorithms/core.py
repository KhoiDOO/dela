import torch
from torch import nn
from typing import *
import argparse


class Algorithm(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()

        self.args = args

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int | float]]:
        raise NotImplementedError()

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, int | float]:
        raise NotImplementedError()
    
    def psnr(self, x_hat: torch.Tensor, err: torch.Tensor):
        return (20 * torch.log10(torch.max(x_hat) / torch.sqrt(err))).item()