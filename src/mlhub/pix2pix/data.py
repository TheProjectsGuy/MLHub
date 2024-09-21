# Datasets in the Pix2Pix work
"""
"""

# %%
import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from natsort import natsorted
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from mlhub.utils import download_and_extract_archive, ex, \
        get_download_dir, set_download_dir, \
        cached_download_and_extract_archive


# %%
_base_url = "https://efrosgans.eecs.berkeley.edu/pix2pix/datasets"
_data_source = {    # [dataset],    All folders in DATA_DIR/dataset/
    # [0 - link, 1 - file, 2 - MD5sum, 3 - train, 4 - val, 
    #  5 - test dir, 6 - jitter, 7 - mirror]
    "cityscapes": [ # Photo to segmentation
        f"{_base_url}/cityscapes.tar.gz",
        "cityscapes.tar.gz",
        "32569df708076e427e9cd02abbca7748",
        "train", None, "val",    # Validation is test
        True, True,    # Jitter and mirror
    ],
    "edges2handbags": [ # Edges to handbag
        f"{_base_url}/edges2handbags.tar.gz",
        "edges2handbags.tar.gz",
        "815592f22fa7f7534a286a25bb73bf03",
        "train", None, "val",    # Validation is test
        False, False,    # No jitter and mirror
    ],
    "edges2shoes": [    # Edges to shoes
        f"{_base_url}/edges2shoes.tar.gz",
        "edges2shoes.tar.gz",
        "71da478b54fedde126e234a2a050bc15",
        "train", None, "val",    # Validation is test
        False, False,    # No jitter and mirror
    ],
    "facades": [    # Building image to segmentation
        f"{_base_url}/facades.tar.gz",
        "facades.tar.gz",
        "e2e25dd517e9d15828416ab329da2854",
        "train", "val", "test",  # Contains all subsets
        True, True,     # Jitter and mirror
    ],
    "maps": [   # Nadir (top) view images to segmented topography view
        f"{_base_url}/maps.tar.gz",
        "maps.tar.gz",
        "e5f65d26eb288457f5ac056deb94d154",
        "train", None, "val",    # Validation is test
        True, True,    # Jitter and mirror
    ],
    "night2day": [  # Night time images to day time image
        f"{_base_url}/night2day.tar.gz",
        "night2day.tar.gz",
        "fd21b1d447a1f927fed41aabedd85baa"
        "train", "test", "val",    # Contains all subsets
        True, True,    # Jitter and mirror
    ],
}


# %%
class Pix2PixDataset(Dataset):
    """
        A dataset wrapper for all Pix2Pix dataset classes from the
        official website: https://efrosgans.eecs.berkeley.edu/pix2pix/datasets
    """
    def __init__(self, dataset_name, use_split="train", 
                swap_st: bool=False):
        self.dataset_name = dataset_name
        self.use_split = use_split
        # Verify if the arguments are correct
        assert self.use_split in ["train", "val", "test"]
        assert self.dataset_name in _data_source.keys(), "Invalid " \
            f" dataset '{self.dataset_name}', should be in " \
            f"{list(_data_source.keys())}"
        if use_split == "train":    # Index of the split
            _ind = 3
        elif use_split == "val":
            _ind = 4
        elif use_split == "test":
            _ind = 5
        else:
            raise ValueError(f"Invalid split '{use_split}', should " \
                f"be in ['train', 'val', 'test']")
        if _data_source[self.dataset_name][_ind] is None:
            raise ValueError(f"Dataset '{self.dataset_name}' does " \
                f"not have a '{use_split}' set")
        # Download and extract the dataset
        # download_and_extract_archive(_data_source[dataset_name][0],
        #         download_root=get_download_dir(),
        #         filename=_data_source[dataset_name][1],
        #         md5=_data_source[dataset_name][2])
        cached_download_and_extract_archive(
                _data_source[dataset_name][0],
                download_root=get_download_dir(),
                flag_root=f"{get_download_dir()}/{dataset_name}",
                filename=_data_source[dataset_name][1],
                md5=_data_source[dataset_name][2])
        self.data_dir = f"{get_download_dir()}/{dataset_name}/" \
                        f"{_data_source[dataset_name][_ind]}"
        self.file_list = [ex(f"{self.data_dir}/{fn}") \
                for fn in natsorted(os.listdir(self.data_dir))]
        # Image transforms
        tfs = [v2.ToDtype(torch.float32, scale=True), 
                v2.Lambda(self.stack_imgs)]
        if _data_source[dataset_name][6]:   # Jitter
            tfs.extend([v2.Resize((286, 286), antialias=True), 
                        v2.RandomCrop((256, 256))])
        if _data_source[dataset_name][7]:   # Mirror
            tfs.append(v2.RandomHorizontalFlip(p=0.5))
        self.tf = v2.Compose(tfs)
        # Swap source and target (domains/image pairs)
        self.swap_st = swap_st
    
    # String representation
    def __repr__(self) -> str:
        ret_str = super().__repr__()
        ret_str += f"\nDataset: {self.dataset_name}"
        ret_str += f"\nSplit: {self.use_split}"
        ret_str += f"\nNumber of items: {len(self)}"
        return ret_str
    
    # Number of items
    def __len__(self):
        return len(self.file_list)
    
    @staticmethod
    def stack_imgs(img: torch.Tensor) -> torch.Tensor:
        """
            Concatenate the image of shape (C, H, 2*W) into a single
            image of shape (2*C, H, W). This takes the second half of
            the image and stacks it on under the first half. The aim
            is to make the torch transforms easier (and consistent
            across the channels). This method also works for batched
            images.
            
            - Input image shape: (B, C, H, 2*W)
            - Output image shape: (B, 2*C, H, W)
        """
        s = img.shape[-1]
        r1 = img[..., :s//2]
        r2 = img[..., s//2:]
        assert r1.shape == r2.shape, f"r1.shape = {r1.shape}, " \
                                    f"r2.shape = {r2.shape}"
        res = torch.cat([r1, r2], dim=-3)   # Stack channels
        return res
    
    # Return a pair
    def __getitem__(self, index):
        img = read_image(self.file_list[index]) # Read image
        st_imgs = self.tf(img)  # Stacked images
        src_img = st_imgs[:3, ...]
        tgt_img = st_imgs[3:, ...]
        if self.swap_st:
            src_img, tgt_img = tgt_img, src_img
        return {
            "source": src_img,
            "target": tgt_img,
            "index": index,
        }


# %%
if __name__ == "__main__":
    set_download_dir("/scratch/mlhub/pix2pix")
    # Show a collage of images
    n_h, n_w = 3, 4 # Grid to show pairs
    ds = Pix2PixDataset("cityscapes", use_split="train")
    # Show figure
    fig = plt.figure(figsize=(n_w*1.8*2.5, n_h*2.5))
    gs = fig.add_gridspec(n_h, n_w)
    v = np.random.choice(len(ds), size=n_h*n_w)
    for i in range(n_h):
        for j in range(n_w):
            ax = fig.add_subplot(gs[i, j])
            imgs = ds[v[i*n_w + j]]
            img = torch.concat([imgs["source"], imgs["target"]], 
                                dim=2)
            img = np.array(v2.functional.to_pil_image(img))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"Pair {v[i*n_w + j] + 1}.jpg")
    fig.set_tight_layout(True)
    plt.show()


# %%
# Experimental section

