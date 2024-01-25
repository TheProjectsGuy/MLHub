# Utilities
"""
"""

# %%
import os
import sys
import torch
import string
import random
import numpy as np
from pathlib import Path
from typing import Optional, Union
from torchvision.datasets.utils import download_and_extract_archive \
    as _download_and_extract_archive


# %%
def ex(x: str):
    """
        Expand a path in 'x'
    """
    return os.path.realpath(os.path.expanduser(x))


# %%
# Download directory
_download_dir = os.getenv("MLHUB_DOWNLOAD_DIR", "/tmp")
# Get download directory
def get_download_dir():
    """
        Get the download directory.
    """
    return ex(_download_dir)
# Set download directory
def set_download_dir(path: str):
    """
        Set the download directory. If directory doesn't exist, it is
        created.
    """
    global _download_dir
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    _download_dir = path
    return _download_dir


# %%
def download_and_extract_archive(url: str, 
        download_root: Optional[str] = None, 
        extract_root: Optional[str] = None, 
        filename: Optional[str] = None, md5: Optional[str] = None,
        remove_finished: bool = False) -> None:
    """
        A wrapper to PyTorch's download and extract function with
        documentation. If the file is already downloaded, then the
        download is not done again (after an MD5 integrity check).
        
        url: Download URL (to obtain the file from)
        download_root: Root folder where downloaded items must be
            stored. If None, then it is inferred from the DOWNLOAD_DIR
            variable. See 'get_download_dir'.
        extract_root: Root folder where the downloaded items are 
            extracted. If None, then it is the same as the 
            'download_root'.
        filename: The filename to use for saving. It is the basename
            of the URL if None.
        md5: The checksum to check the downloaded file against (before
            extracting anything). No check is done if None.
        remove_finished: If True, remove the downloaded file after
            extracting it.
    """
    if download_root is None:
        download_root = get_download_dir()
    _download_and_extract_archive(url, download_root, extract_root, 
                                filename, md5, remove_finished)


# %%
T_IMG = Union[torch.Tensor, np.ndarray]
def norm_img(img: T_IMG, eps=1e-12) -> T_IMG:
    """
        Normalize an image to range [0, 1]
    """
    return (img - img.min()) / (img.max() - img.min() + eps)


# %%
def random_alnum_str(n: int = 4) -> str:
    """
        Generate a random alphanumeric string of length 'n'
    """
    characters = string.ascii_letters + string.digits
    return "".join(random.choices(characters, k=n))
