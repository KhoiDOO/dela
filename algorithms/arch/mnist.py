import torch
from torch import nn

class MnistEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.subnet = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.subnet(x)
    
class MnistDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.subnet = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.subnet(x)

class Mnist_AE_Latent(nn.Module):
    def __init__(self, latent_size) -> None:
        super().__init__()

        self.subnet = nn.Sequential(
            nn.Linear(32 * 7 * 7, latent_size),
            nn.Linear(latent_size, 32 * 7 * 7),
        )
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.subnet(x)
        x = torch.unflatten(x, dim = 1, sizes=(32, 7, 7))
        return x

class Mnist_VAE_Latent(nn.Module):
    def __init__(self, latent_size) -> None:
        super().__init__()

        self.z_mean = nn.Sequential(
            nn.Linear(32 * 7 * 7, latent_size),
            # nn.Linear(16, latent_size),
        )

        self.z_var = nn.Sequential(
            nn.Linear(32 * 7 * 7, latent_size),
            # nn.Linear(16, latent_size),
        )

        self.lat_decode = nn.Linear(latent_size, 32 * 7 * 7)
    
    def forward(self, x):
        x_enc = torch.flatten(x, start_dim=1)
        mu = self.z_mean(x_enc)
        lv = self.z_var(x_enc)
        lat = self.reparam(mu, lv)
        lat_dec = self.lat_decode(lat)
        x_dec = torch.unflatten(lat_dec, dim = 1, sizes=(32, 7, 7))
        return x_dec, mu, lv

    def reparam(self, mu, lv):
        std = torch.exp(0.5 * lv)
        eps = torch.randn_like(std)
        return mu + std * eps