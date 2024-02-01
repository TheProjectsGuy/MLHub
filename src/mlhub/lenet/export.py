# Final external modules (exported for external use)
"""
    Exporting Utilities
    ^^^^^^^^^^^^^^^^^^^^
    
    .. autofunction:: mlhub.lenet.export.download_trained_model
    
"""

# %%
import os
import torch
import hashlib
import numpy as np
from typing import Optional
from torch.hub import download_url_to_file
# MLHub internals
from mlhub.utils import ex, get_download_dir, check_md5
from mlhub.lenet.models import LeNet5


# %%
CKPT_URL = [    # URL, File name, MD5 hash
    "https://www.dropbox.com/scl/fi/q6ru3bj250iq4cuj2qfir/lenet5.pth?rlkey=n4ul5rf6uk6lh2969fp3cnpsh&dl=1",
    "lenet5.pth",
    "31f1b79ebfc97cf2175fd72181008120",
]


# %%
def download_trained_model(ckpt_dir: Optional[str] = None) -> LeNet5:
    """
        Download the trained LeNet-5 model from remote storage and
        load checkpoint. If the checkpoint already exists, then the
        checksum is verified and it's loaded (nothing is downloaded in
        this case).
        
        .. note:: 
            The checkpoint is loaded in ``eval`` mode.
        
        :param ckpt_dir:
            The checkpointing directory (where the ``pth`` file should
            be stored).
        
        :return:    The loaded PyTorch Model
        :rtype:     LeNet5
    """
    if ckpt_dir is None:
        ckpt_dir = f"{get_download_dir()}/checkpoints"
    if not os.path.isdir(ckpt_dir):
        print(f"Creating directory: {ckpt_dir}")
        os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_fname = f"{ex(ckpt_dir)}/{CKPT_URL[1]}"
    if os.path.exists(ckpt_fname):
        print(f"File already exists: {ckpt_fname}")
    else:
        print(f"Downloading checkpoint from {CKPT_URL[0]}")
        download_url_to_file(CKPT_URL[0], ckpt_fname)
        print(f"Download complete: {ckpt_fname}")
    assert check_md5(ckpt_fname, CKPT_URL[2]), "MD5 incorrect"
    model = LeNet5()
    ckpt_data = torch.load(ckpt_fname)
    model.load_state_dict(ckpt_data)
    model.eval()    # Set to evaluation mode
    return model


# %%

