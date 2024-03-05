import torch
from torch import nn

class MnistEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.modules = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, dilation=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.modules(x)
    
class MnistDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.modules = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.modules(x)

class Mnist_AE_Latent(nn.Module):
    def __init__(self, latent_size) -> None:
        super().__init__()

        self.modules = nn.Sequential(
            nn.Linear(32 * 7 * 7, latent_size),
            nn.Linear(latent_size, 32 * 7 * 7),
        )
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.modules(x)
        x = torch.unflatten(x, dim = 1, sizes=(32, 7, 7))
        return x