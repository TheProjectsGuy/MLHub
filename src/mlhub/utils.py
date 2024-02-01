# Utilities
"""
    The ``utils`` module contains common utilities for MLHub.
    
    Configurations
    ---------------
    
    Download Directory
    ^^^^^^^^^^^^^^^^^^^
    
    This is the directory where all the files (datasets, checkpoints,
    etc.) are downloaded. It's used for dataloaders, saving 
    checkpoints during training, loading checkpoints for models, etc.
    
    It is ``/tmp`` by default. However, it can be set through the 
    following means
    
    -   By setting the environment variable ``MLHUB_DOWNLOAD_DIR`` to
        the desired directory.
    -   Using the :py:func:`mlhub.utils.set_download_dir` function.
    
    You can get the current download directory using the
    :py:func:`mlhub.utils.get_download_dir` function. The value of
    this variable is referred as ``DOWNLOAD_DIR`` in the docs.
    
    .. autofunction:: mlhub.utils.get_download_dir
    .. autofunction:: mlhub.utils.set_download_dir
    
    File Management
    ----------------
    
    .. autofunction:: mlhub.utils.ex
    .. autofunction:: mlhub.utils.download_and_extract_archive
    .. autofunction:: mlhub.utils.check_md5
    
    Images
    -------
    
    .. autofunction:: mlhub.utils.norm_img
    
    Miscellaneous
    --------------
    
    .. autofunction:: mlhub.utils.random_alnum_str
    
"""

# %%
import os
import sys
import torch
import string
import random
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional, Union
from torchvision.datasets.utils import download_and_extract_archive \
    as _download_and_extract_archive


# %%
# ----------- File Management -----------
# Expand the path fully
def ex(x: str) -> str:
    r"""
        Expand a path fully (to realpath). Also expands ``~`` (tilde)
        to home.
        
        :param x:   A path
        
        :return:    A fully resolved (absolute) path
    """
    return os.path.realpath(os.path.expanduser(x))


# Download and extract an file from a URL
def download_and_extract_archive(url: str, 
        download_root: Optional[str] = None, 
        extract_root: Optional[str] = None, 
        filename: Optional[str] = None, md5: Optional[str] = None,
        remove_finished: bool = False) -> None:
    r"""
        A wrapper to PyTorch's download and extract function with
        documentation. If the file is already downloaded, then the
        download is not done again (after an MD5 integrity check).
        
        :param url:     The download URL (to obtain the file from)
        :param download_root:
                Root folder where downloaded items must be stored. 
                If None, then it is inferred from the function 
                :py:func:`mlhub.utils.get_download_dir`.
        :param extract_root:
                Root folder where the downloaded items are extracted. 
                If None, then it is the same as the ``download_root``
        :param filename:    
                The filename to use for saving. It is the basename of 
                the URL if None.
        :param md5: 
                The checksum to check the downloaded file against 
                (before extracting anything). No check is done if 
                ``None``.
        :param remove_finished: 
                If True, remove the downloaded file after extracting 
                it.
    """
    if download_root is None:
        download_root = get_download_dir()
    _download_and_extract_archive(url, download_root, extract_root, 
                                filename, md5, remove_finished)


# Check the MD5 checksum of a file
def check_md5(file: str, true_md5: Optional[str] = None) \
        -> Union[str, bool]:
    """
        Returns the MD5 checksum of the given file
        
        :param file:     The file to check (should exist)
        :param true_md5: 
            The true MD5 checksum of the file. If None, then the 
            checksum is not checked and the function returns the MD5
            checksum of the file. If an expected (true) hash is passed
            then the function returns ``True`` if the MD5 matches (
            ``False`` otherwise)
        
        :return:
            The MD5 checksum of the file if ``true_md5`` is None. Else
            a bool comparing ``true_md5`` with the MD5 of ``file``.
    """
    if not os.path.isfile(file):
        raise FileNotFoundError(file)
    # Get hash of file
    with open(ex(file), "rb") as f:
        md5_hash = hashlib.md5(f.read()).hexdigest()
    if true_md5 is not None:
        return md5_hash == true_md5
    return md5_hash


# %%
# ----------- Download directory -----------
_download_dir = os.getenv("MLHUB_DOWNLOAD_DIR", "/tmp")

# Get download directory
def get_download_dir() -> str:
    r"""
        Get the download directory (as absolute/resolved path). Only
        use :py:func:`mlhub.utils.set_download_dir` to set the
        download directory.
        
        :return:    The fully resolved download directory
    """
    return ex(_download_dir)

# Set download directory
def set_download_dir(path: str) -> str:
    r"""
        Set the download directory. If directory doesn't exist, it is
        created. Use :py:func:`mlhub.utils.get_download_dir` to get
        the current download directory.
        
        .. note::
            By default, the download directory is set by the 
            environment variable ``MLHUB_DOWNLOAD_DIR``. If it's not 
            set, then the default is ``/tmp``.
        
        :param path:   The download directory
        
        :return:       The download directory
    """
    global _download_dir
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    _download_dir = path
    return _download_dir


# %%
# ----------- Images -----------
# Normalize an image
T_IMG = Union[torch.Tensor, np.ndarray]
def norm_img(img: T_IMG, eps: float = 1e-12) -> T_IMG:
    r"""
        Normalize an image (uniformly map [min, max]) to range [0, 1].
        
        :param img:     The image to normalize. This is not modified.
        :param eps:     A small value to avoid division by zero
        
        :return:        The normalized image.
    """
    return (img - img.min()) / (img.max() - img.min() + eps)


# %%
def random_alnum_str(n: int = 4) -> str:
    """
        Generate a random alphanumeric string of length ``n``.
        Characters could be repeated.
        
        :param n:   The length of alphanumeric string
    """
    characters = string.ascii_letters + string.digits
    return "".join(random.choices(characters, k=n))
