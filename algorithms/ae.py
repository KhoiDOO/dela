from argparse import Namespace
from typing import Dict, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from .core import Algorithm

class AE(Algorithm):
    def __init__(self, args: Namespace) -> Algorithm:
        super().__init__(args)

        self.args = args

        if self.args.ds == 'mnist':
            from .arch import Mnist_AE_Latent
            self.latent = Mnist_AE_Latent(latent_size=self.args.ldim)
        elif self.args.ds == 'cifar10':
            from .arch import Cifar10_AE_Latent
            self.latent = Cifar10_AE_Latent(latent_size=self.args.ldim)
        elif self.args.ds == 'cinic10':
            from .arch import Cinic10_AE_Latent
            self.latent = Cinic10_AE_Latent(latent_size=self.args.ldim)
        else:
            raise Exception(f"The architecture for dataset {self.args.ds} is not currently supported in method AutoEncoder")
    
    def forward(self, x: Tensor, y: Tensor = None) -> Tuple[Tensor, Dict[str, int | float]]:
        x_enc = self.encoder(x)
        lat = self.latent(x_enc)
        x_hat = self.decoder(lat)

        loss = F.mse_loss(x, x_hat)

        return loss, {"train/rec_loss": loss.item(), "train/psnr" : self.psnr(x_hat=x_hat, err=loss)}
    
    def evaluate(self, x: Tensor, y: Tensor = None) -> Dict[str, int | float]:
        with torch.no_grad():
            x_enc = self.encoder(x)
            lat = self.latent(x_enc)
            x_hat = self.decoder(lat)

            loss = F.mse_loss(x, x_hat)
        
        return loss, {"valid/rec_loss": loss.item(), "valid/psnr" : self.psnr(x_hat=x_hat, err=loss)}