from argparse import Namespace
from typing import Dict, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from .core import Algorithm


class VAE(Algorithm):
    def __init__(self, args: Namespace) -> Algorithm:
        super().__init__(args)

        self.args = args

        if self.args.ds == 'mnist':
            from .arch import Mnist_VAE_Latent
            self.latent = Mnist_VAE_Latent(latent_size=self.args.ldim)
        else:
            raise Exception(f"The architecture for dataset {self.args.ds} is not currently supported in method AutoEncoder")
    
    def forward(self, x: Tensor, y: Tensor = None) -> Tuple[Tensor, Dict[str, int | float]]:
        x_enc = self.encoder(x)
        lat, mu, lv = self.latent(x_enc)
        x_hat = self.decoder(lat)

        rec_loss = F.mse_loss(x, x_hat)
        kld_loss = self.gaussian_kls(mu, lv)

        loss = rec_loss + kld_loss

        return loss, {
            "train/loss" : loss.item(), 
            "train/rec_loss": rec_loss.item(), 
            "train/kld_loss": kld_loss.item(), 
            "train/psnr" : self.psnr(x_hat=x_hat, err=rec_loss)
        }
    
    def evaluate(self, x: Tensor, y: Tensor = None) -> Tuple[Tensor, Dict[str, int | float]]:
        with torch.no_grad():
            x_enc = self.encoder(x)
            lat, mu, lv = self.latent(x_enc)
            x_hat = self.decoder(lat)

            rec_loss = F.mse_loss(x, x_hat)
            kld_loss = self.gaussian_kls(mu, lv)

            loss = rec_loss + kld_loss

        return loss, {
            "valid/loss" : loss.item(), 
            "valid/rec_loss": rec_loss.item(), 
            "valid/kld_loss": kld_loss.item(), 
            "valid/psnr" : self.psnr(x_hat=x_hat, err=rec_loss)
        }

    @staticmethod
    def gaussian_kls(mu, logvar):
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return torch.mean(kld_loss)