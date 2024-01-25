# %%
import os
import sys
import tyro
import time
import torch
import traceback
import numpy as np
import torch.nn as nn
from torch import device
from tqdm.auto import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Optional, Tuple
# MLHub internals
from mlhub.utils import get_download_dir, set_download_dir, ex, \
        norm_img, random_alnum_str
from .models import LeNet5
from .data import MNISTDataset
from .utils import error_rate, model_output_to_labels, test


# %%
@dataclass
class LocalArgs:
    # Batch size to use for the training process
    train_batch_size: int = 32
    # Learning rate to use for training process
    train_learning_rate: float = 1e-2
    # Number of epochs to train for
    train_epochs: int = 20
    # Parameter 'j' for the training loss function
    train_fn_j: float = 0.01
    # Directory where to store the trained model (checkpoint)
    ckpt_dir: str = "./model"
    # Directory where to store the downloaded dataset (None = default)
    download_dir: Optional[str] = None


# %%
class TrainingLoss(nn.Module):
    def __init__(self, j: float = 0.01) -> None:
        super().__init__()
        self.j = j
    
    def forward(self, model_output, target_output):
        bs = len(target_output)
        assert bs == model_output.shape[0]
        loss = 0
        for i in range(bs):
            loss += model_output[i, int(target_output[i])]
            loss += torch.log(np.exp(-self.j) \
                            + torch.sum(torch.exp(-model_output[i])))
        loss /= bs
        return loss


# %%
class Trainer:
    """
        Trainer for LeNet-5
    """
    def __init__(self, batch_size: int = 32, 
                learning_rate: float = 1e-2,
                training_loss: Optional[Callable] = None,
                ckpt_dir: str = None,
                device: Optional[device] = None) -> None:
        # Datasets
        self._batch_size = batch_size
        self.train_dataset = MNISTDataset(train=True)
        self.train_len = len(self.train_dataset)
        self.train_dataloader = DataLoader(self.train_dataset, 
                batch_size=self._batch_size, shuffle=True)
        self.test_dataset = MNISTDataset(train=False)
        self.test_len = len(self.test_dataset)
        self.test_dataloader = DataLoader(self.test_dataset,
                batch_size=self._batch_size)
        # Model
        self.model = LeNet5()
        # Check for CUDA
        self.device = device
        if self.device is None and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using device: {torch.cuda.get_device_name(0)}")
            self.model.cuda()
        elif self.device is None:
            self.device = torch.device("cpu")
            print(f"Training on CPU")
        else:
            self.device = torch.device(self.device)
            print(f"Using device: {self.device}")
        # Optimizer
        self._learning_rate = learning_rate
        self.optimizer = optim.SGD(self.model.parameters(), 
                lr=self._learning_rate)
        # Loss
        if training_loss is not None:
            self.training_loss = training_loss
        else:
            self.training_loss = TrainingLoss()
        # Checkpoint directory (and logging)
        self.ckpt_dir = ckpt_dir
        if self.ckpt_dir is None:
            self.writer = None
        else:
            tb_logs_dir = f"{self.ckpt_dir}/runs/"\
                f"{random_alnum_str(4)}-"\
                f"{time.strftime(r'%Y_%m_%dT%H_%M_%S')}"
            self.writer = SummaryWriter(ex(tb_logs_dir))
    
    # Train model for a single epoch
    def _train_epoch(self, curr_epoch: int = 0):
        self.model.train()
        for batch, tr_sample in enumerate(self.train_dataloader):
            # Resolve input image and labels (target)
            tr_input, tr_target = tr_sample
            tr_input = tr_input.to(self.device)
            tr_target = tr_target.to(self.device)
            # Compute prediction error
            model_output = self.model(tr_input)
            loss = self.training_loss(model_output, tr_target)
            # Backpropagation
            self.optimizer.zero_grad()
            #
            # FIXME: Ideally, 'retain_graph=True' should not be used.
            #   For some reason, this doesn't work without it. It 
            #   keeps throwing the error `RuntimeError: Trying to 
            #   backward through the graph a second time`.
            #   I guess the network has some parameter that's 
            #   accumulating gradients (which it shouldn't). To debug,
            #   I'd suggest trying out:
            #   - Replace the model with something simple
            #   - Different (probably inbuilt) loss function
            #
            #   However, using this doesn't cause the VRAM use to 
            #   explode and still leads to stable learning.
            loss.backward(retain_graph=True)
            self.optimizer.step()
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(tr_input)
                if self.writer is not None:
                    self.writer.add_scalar(
                        f"epoch_{curr_epoch+1}/loss", loss, current)
                else:
                    print(f"Loss: {loss:<15f} "\
                            f"[{current:>5d}/{self.train_len:>5d}]")
        return loss.item()
    
    # Test the model through the test set
    @torch.no_grad()
    def test(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        er, model_preds, test_preds = test(self.test_dataloader, 
                self.model, self.device)
        return er, model_preds, test_preds
    
    # Checkpoint model
    def _checkpoint(self, ckpt_fname: str, model_only: bool = True, 
                **kwargs):
        """
            Checkpoint the model and additional keyword arguments.
            The function doesn't check for 'ckpt_dir' value.
            If 'model_only' is False, then optimizer is also stored.
        """
        save_dict = {
            "model_state_dict": self.model.state_dict(),
        }
        if not model_only:
            save_dict.update({
                "optimizer_state_dict": self.optimizer.state_dict(),
            })
        if kwargs is not None:
            save_dict.update(kwargs)
        torch.save(save_dict, ckpt_fname)
    
    # Train the model for the specified number of epochs
    def train(self, num_epochs: int = 20) -> LeNet5:
        best_test_er, best_test_epoch = None, None
        for i in tqdm(range(num_epochs)):
            loss = self._train_epoch(i)
            er, _, _ = self.test()
            if self.writer is not None:
                self.writer.add_scalar(f"train/loss", loss, i)
                self.writer.add_scalar(f"test/error", er, i)
            if best_test_er is None or er < best_test_er:
                best_test_epoch, best_test_er = i, er
                if self.ckpt_dir is not None:
                    self._checkpoint(
                        f"{self.ckpt_dir}/checkpoint_epoch_{i}.pt",
                        model_only=False, loss=loss, epoch=i, 
                        test_error=er, device=self.device)
        print("Training done!")
        return self.model, (best_test_epoch, best_test_er)


# %%
def main(args: LocalArgs):
    if args.download_dir is not None:
        set_download_dir(args.download_dir)
    print(f"Dataset will be downloaded at: {get_download_dir()}")
    ckpt_dir = ex(args.ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
        print(f"Created directory: {ckpt_dir}")
    else:
        print("WARNING: Checkpoint directory already exists. " \
                "It is possible that things will be overwritten.")
    trainer = Trainer(args.train_batch_size, 
                        args.train_learning_rate, 
                        TrainingLoss(args.train_fn_j), ckpt_dir)
    _, (ep, er) = trainer.train(args.train_epochs)
    print(f"Best test error rate {er} on epoch {ep}")
    src_ckpt_file = f"{ckpt_dir}/checkpoint_epoch_{ep}.pt"
    model = LeNet5()
    best_ckpt = torch.load(src_ckpt_file)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval().cpu()
    res_ckpt_file = f"{ex(ckpt_dir)}/checkpoint.pth"
    torch.save(model.state_dict(), res_ckpt_file)
    print(f"Checkpoint created at '{res_ckpt_file}'")
    # Test the model in the end
    model = LeNet5()
    model.load_state_dict(torch.load(res_ckpt_file))
    model.eval()
    model.to(trainer.device)
    er, _, _ = test(
        trainer.test_dataloader, model, trainer.device)
    print(f"Final test gives error rate: {er}")


# %%
if __name__ == "__main__" and "ipykernel" not in sys.argv[0]:
    try:
        start_time = time.time()
        args = tyro.cli(LocalArgs)
        print(f"Arguments: {args}")
        main(args)
        end_time = time.time()
        print(f"Total time taken: {end_time - start_time:.2f}s")
    except SystemExit as exc:
        print(f"System exit: {exc}")
    except:
        traceback.print_exc()
    exit(0)


# %%
# Experimental section


# %%
# main(LocalArgs(ckpt_dir="/scratch/mlhub/checkpoints/lenet5", 
#                 download_dir="/scratch/mlhub"))

# # %%
# set_download_dir("/scratch/mlhub")
# trainer = Trainer(32, 0.01, TrainingLoss(), 
#                 "/scratch/mlhub/checkpoints/lenet5")

# # %%
# model, _ = trainer.train(2)

# # %%
# er, _, _, = test(trainer.test_dataloader, model, trainer.device)
# print(f"Error rate: {er}")

# # %%
# model.cpu().eval()

# # %%
# er, _, _, = test(trainer.test_dataloader, model, "cpu")
# print(f"Error rate: {er}")

# # %%
# torch.save(model.state_dict(), 
#         "/scratch/mlhub/checkpoints/lenet5/checkpoint.pth")

# # %%
# m2 = LeNet5()

# # %%
# m2.load_state_dict(
#     torch.load("/scratch/mlhub/checkpoints/lenet5/checkpoint.pth"))

# # %%
# er, _, _, = test(trainer.test_dataloader, m2, "cpu")
# print(f"Error rate: {er}")

# # %%
# er, _, _, = test(trainer.test_dataloader, model, "cpu")
# print(f"Error rate: {er}")

# # %%
# m1_prms = [p for p in model.parameters()]
# m2_prms = [p for p in m2.parameters()]

# %%
