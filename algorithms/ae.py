from argparse import Namespace
import os, sys
from typing import Dict, Tuple

from torch import Tensor
from .core import Algorithm

class AE(Algorithm):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)

        self.args = args

        if self.args.ds == 'mnist':
            from models import Mnist_AE_Latent
            self.latent = Mnist_AE_Latent()
        else:
            raise Exception(f"The architecture for dataset {self.args.ds} is not currently supported in method AutoEncoder")
    
    def forward(self, x: Tensor, y: Tensor = None) -> Tuple[Tensor, Dict[str, int | float]]:
        return super().forward(x, y)