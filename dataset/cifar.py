from typing import Callable
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
from typing import *


class CusCIFAR10(CIFAR10):
    def __init__(self, 
                 root: str = "/".join(__file__.split("/")[:-1]) + "/source", 
                 train: bool = True, 
                 transform: Callable[..., Any] | None = transforms.Compose([transforms.ToTensor()]), 
                 target_transform: Callable[..., Any] | None = None, 
                 download: bool = True) -> CIFAR10:
        super().__init__(root, train, transform, target_transform, download)

class CusCIFAR100(CIFAR100):
    def __init__(self, 
                 root: str = "/".join(__file__.split("/")[:-1]) + "/source", 
                 train: bool = True, 
                 transform: Callable[..., Any] | None = transforms.Compose([transforms.ToTensor()]), 
                 target_transform: Callable[..., Any] | None = None, 
                 download: bool = True) -> CIFAR100:
        super().__init__(root, train, transform, target_transform, download)