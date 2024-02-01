# Testing LeNet
"""
    This is for testing the trained model. This is mainly internal
    to the LeNet sub-module. The trained LeNet-5 was tested using
    
    .. code-block:: bash
    
        python -m mlhub.lenet.test \\
            --ckpt-dir /scratch/mlhub/checkpoints/lenet5 \\
            --download-dir /scratch/mlhub
    
    It basically runs the model through the test set, reports the test
    error, and allows you to sample results (view as matplotlib 
    figures).
"""

# %%
import os
import sys
import time
import tyro
import torch
import traceback
import numpy as np
from torch import nn
from typing import Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from torch.utils.data import DataLoader
# MLHub internals
from .models import LeNet5
from .data import MNISTDataset
from .utils import test, model_output_to_labels, \
        model_output_to_multi_labels
from mlhub.utils import set_download_dir, get_download_dir, ex, \
        norm_img


# %%
@dataclass
class LocalArgs:
    # Dataset directory (where it's downloaded). None = default
    download_dir: Optional[str] = None
    # Directory where the trained model (checkpoint) is stored
    ckpt_dir: str = "./model"
    # Batch size for the test dataset
    test_batch_size: int = 32


# %%
def main(args: LocalArgs):
    """
        Main function
    """
    if args.download_dir is not None:
        set_download_dir(args.download_dir)
    print(f"Dataset will be downloaded at: {get_download_dir()}")
    ckpt_file = f"{ex(args.ckpt_dir)}/checkpoint.pth"
    if not os.path.isfile(ckpt_file):
        raise FileNotFoundError("No checkpoint found. "\
                "Please train the model first.")
    # Model
    model = LeNet5()
    # Load checkpoint and set to evaluation mode
    checkpoint = torch.load(ckpt_file)
    model.load_state_dict(checkpoint)
    model.eval()
    # Test dataset
    test_dataset = MNISTDataset(train=False)
    test_dataloader = DataLoader(test_dataset,
                                batch_size=args.test_batch_size)
    device = "cpu"
    if torch.cuda.is_available():
        model.cuda()
        print(f"Using device: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("Using CPU")
    device = torch.device(device)
    # Get inference
    er, model_preds, test_preds = test(test_dataloader, model)
    print(f"Error rate on test set: {er}")
    model_preds = model_preds.cpu().numpy()
    test_preds = test_preds.cpu().numpy()
    inds = np.arange(len(model_preds))[model_preds != test_preds]
    print(f"Wrong/Mistaken indices: {inds.tolist()}")
    _prompt_str = "Enter index to visualize (blank = end): "
    viz_ind = input(_prompt_str)
    while viz_ind != "":
        viz_ind = int(viz_ind)
        top_n = 3   # If wrong output, show top_n predictions too
        with torch.no_grad():
            test_sample = test_dataset[viz_ind]
            test_img, test_label = test_sample[0], test_sample[1]
            test_img = test_img.to(device)
            test_label = test_label
            model_output = model(test_img)
            test_pred = model_output_to_labels(model_output)
            test_pred = int(test_pred)
            if test_label != test_pred:
                tn = model_output_to_multi_labels(model_output, top_n)
                tn = tn.cpu().numpy()
                print(f"\tTop {top_n} predictions: {tn}")
                print(f"\tModel output (for debugging): "\
                        f"{model_output.cpu().numpy()}")
        plt.imshow(norm_img(test_img.cpu()).numpy()[0])
        plt.title(f"Target: {test_label}, Pred: {test_pred}")
        plt.show()
        viz_ind = input(_prompt_str)


# %%
if __name__ == "__main__" and "ipykernel" not in sys.argv[0]:
    try:
        start_time = time.time()
        args = tyro.cli(LocalArgs)
        print(f"Arguments: {args}")
        main(args)
        end_time = time.time()
        print(f"Total time: {end_time - start_time:2f}s")
    except SystemExit as exc:
        print(f"System exit: {exc}")
    except:
        traceback.print_exc()
    exit(0)

# %%
