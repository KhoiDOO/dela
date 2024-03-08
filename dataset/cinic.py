from typing import Callable
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
import torchvision.transforms as transforms
from typing import *

class CINIC10(DatasetFolder):
    def __init__(self, 
                 split: str = 'train', 
                 loader: Callable[[str], Any] = default_loader, 
                 extensions: Tuple[str, ...] | None = IMG_EXTENSIONS, 
                 transform: Callable[..., Any] | None = transforms.Compose(
                        [
                            transforms.RandomAffine(degrees=(-1, 1), translate=(0.1, 0.1)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])
                        ]
                    ), 
                 target_transform: Callable[..., Any] | None = None, 
                 is_valid_file: Callable[[str], bool] | None = None) -> None:
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f"split must be one of ['train', 'valid', 'test'] in CINIC10, but found {split} instead")
        root: str = "/".join(__file__.split("/")[:-1]) + f"/source/cinic10/{split}"
        super().__init__(
            root, 
            loader, 
            extensions, 
            transform, 
            target_transform, 
            is_valid_file
        )
    
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target