import torch
from torch import nn
from typing import *
import argparse

class Algorithm(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()

        self.args = args

        self.prepare_model()
    
    def prepare_model(self):
        if self.args.ds == 'mnist':
            from .arch import MnistEncoder, MnistDecoder
            self.encoder = MnistEncoder()
            self.decoder = MnistDecoder()
        elif self.args.ds == 'cifar10':
            from .arch import Cifar10Encoder, Cifar10Decoder
            self.encoder = Cifar10Encoder()
            self.decoder = Cifar10Decoder()
        elif self.args.ds == 'cinic10':
            from .arch import Cinic10Encoder, Cinic10Decoder
            self.encoder = Cinic10Encoder()
            self.decoder = Cinic10Decoder()
        else:
            raise Exception(f"The architecture for dataset {self.args.ds} is not currently supported")

    def forward(self, x: torch.Tensor, y: torch.Tensor=None) -> Tuple[torch.Tensor, Dict[str, int | float]]:
        raise NotImplementedError()

    def evaluate(self, x: torch.Tensor, y: torch.Tensor=None) -> Dict[str, int | float]:
        raise NotImplementedError()
    
    @staticmethod
    def psnr(x_hat: torch.Tensor, err: torch.Tensor):
        return (20 * torch.log10(torch.max(x_hat) / torch.sqrt(err))).item()