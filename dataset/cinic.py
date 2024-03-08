from typing import Callable
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
from typing import *


class CINIC10(VisionDataset):
    def __init__(self, 
                 split: str = 'train', 
                 transforms: Callable[..., Any] | None = None, 
                 transform: Callable[..., Any] | None = transforms.Compose(
                        [
                            transforms.RandomAffine(degrees=(-1, 1), translate=(0.1, 0.1)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])
                        ]
                    ), 
                 target_transform: Callable[..., Any] | None = None
                 ) -> VisionDataset:
        
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f"split must be one of ['train', 'valid', 'test'] in CINIC10, but found {split} instead")
        root: str = "/".join(__file__.split("/")[:-1]) + f"/source/cinic10/{split}",
        super().__init__(root, transforms, transform, target_transform)