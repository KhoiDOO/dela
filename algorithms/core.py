import torch
from torch import nn
from typing import *
import argparse

class Algorithm(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()

        self.args = args
    
    def prepare_model(self):
        if self.args.ds == 'mnist':
            from .arch.mnist import MnistEncoder, MnistDecoder
            self.encoder = MnistEncoder()
            self.decoder = MnistDecoder()
        else:
            raise Exception(f"The architecture for dataset {self.args.ds} is not currently supported")

    def forward(self, x: torch.Tensor, y: torch.Tensor=None) -> Tuple[torch.Tensor, Dict[str, int | float]]:
        raise NotImplementedError()

    def evaluate(self, x: torch.Tensor, y: torch.Tensor=None) -> Dict[str, int | float]:
        raise NotImplementedError()
    
    @staticmethod
    def psnr(x_hat: torch.Tensor, err: torch.Tensor):
        return (20 * torch.log10(torch.max(x_hat) / torch.sqrt(err))).item()