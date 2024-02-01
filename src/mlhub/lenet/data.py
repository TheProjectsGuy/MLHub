# Dataset utilities and dataloaders for LeNet
"""
    Datasets and dataloaders for the `MNIST dataset <http://yann.lecun.com/exdb/mnist/index.html>`__.
    
    .. autoclass:: mlhub.lenet.data.MNISTDataset
        :members:
        :special-members: __len__, __init__, __getitem__
"""

# %%
import os
import sys
import torch
import idx2numpy
import numpy as np
import einops as ein
import matplotlib.pyplot as plt
from typing import Optional, Union
from collections.abc import Callable
from torch.utils.data import Dataset
from torchvision.transforms import v2 as tvf
# MLHub internals
from mlhub.utils import download_and_extract_archive, ex, \
    get_download_dir


# %%
_data_source = {    # ["Link", "MD5"]
    "train-images": [
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        "train-images-idx3-ubyte"
    ],
    "train-labels": [
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "d53e105ee54ea40749a09fcbcd1e9432",
        "train-labels-idx1-ubyte"
    ],
    "test-images": [
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "9fb629c4189551a2d022fa330f9573f3",
        "t10k-images-idx3-ubyte"
    ],
    "test-labels": [
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
        "ec29112dd5afa0611ce80d1b7f02629c",
        "t10k-labels-idx1-ubyte"
    ]
}


# %%
_default_transform = tvf.Compose([
    tvf.ToImage(), tvf.ToDtype(torch.float32, scale=True),
    tvf.Pad(2), # Convert (28, 28) to (32, 32)
    # Final mean = 0 and var = 1 (approx). Values calculated over
    #   entire training set
    tvf.Normalize(mean=[0.1037], std=[0.3081])
])


# %%
T1 = Union[str, Callable, None]
T2 = tuple[Union[torch.Tensor, np.ndarray], int]
# MNIST Dataset class
class MNISTDataset(Dataset):
    """
        MNIST Dataset class. A child class of 
        `torch.utils.data.Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`__.
    """
    def __init__(self, download_root: Optional[str] = None, 
                train: bool = True, 
                transform: T1 = "default", 
                target_transform: T1 = None) -> None:
        """
            :param download_root:
                The root directory where to download the files. If
                ``None``, then it is ``DOWNLOAD_DIR/mnist``.
            :param train:
                If True, use the training set, else use the test set.
            :param transform:
                Transformation to apply to the images. No 
                transformation is applied if ``None``. If ``default``,
                then
                
                1.  Convert to PyTorch tensor
                2.  Zero pad (28, 28) image to (32, 32) shape
                3.  Normalize input (so that mean and std are approx
                    0 and 1, respectively)
                
                If some custom transform is given, it should be 
                callable. It's best to give torchvision transforms.
            :param target_transform:
                Transformation to apply to the labels. ``None`` means
                no transform.
        """
        super().__init__()
        self.data: dict[str, np.ndarray] = {}  # Complete data
        # Download files
        if download_root is None:
            self.download_root = f"{get_download_dir()}/mnist"
        else:
            self.download_root = ex(download_root)
        self.train = train
        if self.train:
            self.data_parts = ["train-images", "train-labels"]
        else:
            self.data_parts = ["test-images", "test-labels"]
        # Data parts
        for part in self.data_parts:
            print(f"Processing segment '{part}' of MNIST dataset")
            download_and_extract_archive(_data_source[part][0],
                    self.download_root, md5=_data_source[part][1])
            # Load to array
            self.data[part]: np.ndarray = idx2numpy.convert_from_file(
                    f"{self.download_root}/{_data_source[part][2]}")
        # Register transforms
        self.transform = transform
        self.target_transform = target_transform
        if self.transform == "default":
            self.transform = _default_transform
    
    def __len__(self) -> int:
        """
            Returns the number of samples in the MNIST (train or test)
            split.
        """
        return len(self.data[self.data_parts[0]])
    
    def __getitem__(self, idx: int) -> T2:
        """
            Returns a sample at the index ``idx``
            
            :param idx: The index
            :returns:   A tuple containing the ``(image, label)``
        """
        img, label = self.data[self.data_parts[0]][idx], \
                self.data[self.data_parts[1]][idx]
        img: np.ndarray = ein.rearrange(img, "h w -> h w 1")
        if self.transform is not None:
            img = self.transform(img.copy())
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label


# %%
