from typing import Callable
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from typing import *


class CusMNIST(MNIST):
    def __init__(self, 
                 root: str = "/".join(__file__.split("/")[:-1]) + "/source", 
                 train: bool = True, 
                 transform: Callable[..., Any] | None = transforms.Compose([transforms.ToTensor()]), 
                 target_transform: Callable[..., Any] | None = None, 
                 download: bool = True) -> MNIST:
        super().__init__(root, train, transform, target_transform, download)