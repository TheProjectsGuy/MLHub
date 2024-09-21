# Training the Pix2Pix network with loss implementation
"""
    Main training
"""

# %%
import os
import sys
import yaml
import tyro
import time
import torch
import numpy as np
from torch import nn
import einops as ein
from PIL import Image
from torch import optim
from tqdm.auto import tqdm
from datetime import datetime
from typing import Optional, Literal
from matplotlib import pyplot as plt
from torch.nn import functional as F
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
# Internal imports
from mlhub.utils import get_download_dir, set_download_dir, \
        random_alnum_str, norm_img, ex
# ------ Development imports ------
# Data
from mlhub.pix2pix.data import Pix2PixDataset
from mlhub.pix2pix.data import _data_source
# Generator and discriminator models
from mlhub.pix2pix.models import UNetGenerator, EncDecGenerator, \
        Discriminator


# %%
@dataclass
class TrainerArgs:
    # Learning rate for the generator
    gen_lr: float = 2e-4
    # Learning rate for the discriminator
    disc_lr: float = 1e-4
    # Batch size (for discriminator training)
    bs_disc: int = 20
    # Batch size (for generator training)
    bs_gen: int = 20
    # Set to True to use tensorboard and False to turn it off
    use_tb: bool = True
    # Tensorboard directory (set `use_tb` for this to work)
    tb_dir: str = "/scratch/tb_runs/pix2pix"
    # Weight of the L1 loss for generator (lambda in paper)
    gen_l1_w: float = 50

# Arguments
@dataclass
class LocalArgs:
    # Trainer class
    train: TrainerArgs
    # Dataset to use
    ds: str
    # Swap source and target
    swap_st: bool = False
    # Number of epochs
    num_epochs: int = 10
    # Use GPU if True (and available), else use CPU
    use_gpu: bool = True
    # Download directory (for datasets and output checkpoints)
    download_dir: str = get_download_dir()


# %%
class Trainer:
    def __init__(self, ds: Dataset, gen: nn.Module, 
                disc: nn.Module, device: torch.device = "cpu", 
                train_params: Optional[TrainerArgs] = None,
                dl_test: Optional[DataLoader] = None) -> None:
        if train_params is None:
            train_params = TrainerArgs()
        self.ds = ds
        self.dl_disc = DataLoader(ds, batch_size=train_params.bs_disc,
                                shuffle=True)
        self.dl_gen = DataLoader(ds, batch_size=train_params.bs_gen, 
                                shuffle=True)
        self.device = torch.device(device)
        self.generator = gen.to(self.device)
        self.discriminator = disc.to(self.device)
        self.optim_gen = optim.Adam(gen.parameters(), 
                lr=train_params.gen_lr, betas=(0.5, 0.999))
        self.optim_disc = optim.Adam(disc.parameters(), 
                lr=train_params.disc_lr, betas=(0.5, 0.999))
        self.trainer_id = random_alnum_str(4)
        if train_params.use_tb:
            tb_dir = f"{train_params.tb_dir}/"\
                f"{datetime.now().strftime(r'%Y-%m-%dT%H-%M-%S')}_"\
                f"{self.trainer_id}"    # Timestamp + ID
            self.tb_writer = SummaryWriter(tb_dir)
            print(f"Tensorboard will be saved at: {tb_dir}")
        else:
            self.tb_writer = None
            print(f"Not using tensorboard")
        self.dl_test = dl_test  # Test (validation) set
        self.train_params = train_params
        print(f"Started trainer with ID: {self.trainer_id}")
    
    def _set_disc_trainable(self, state: bool = True):
        for param in self.discriminator.parameters():
            param.requires_grad = state
    
    def loss_desc_gan(self, disc_gen, disc_true):
        disc_outputs = torch.cat([disc_gen, disc_true]) # (2*b)
        disc_out_classes = torch.cat([  # 0 (generated), 1 (true)
                torch.zeros(disc_gen.size()),
                torch.ones(disc_true.size())])  # (2*b) shape
        disc_out_classes = disc_out_classes.to(self.device)
        loss_disc_gan = F.binary_cross_entropy(disc_outputs, 
                    disc_out_classes)   # Discriminator GAN loss
        return loss_disc_gan
    
    def loss_gen_gan_l1(self, gen_imgs, disc_gen_tr, target_imgs):
        loss_gen_l1 = F.l1_loss(gen_imgs, target_imgs)
        # loss_gen_gan = torch.log(1 - disc_gen_tr).mean() # < 0
        loss_gen_gan = -torch.log(disc_gen_tr).mean() # < 0
        loss_gen = loss_gen_gan \
                + self.train_params.gen_l1_w * loss_gen_l1
        return loss_gen, loss_gen_gan, loss_gen_l1
    
    def _train_epoch(self, curr_epoch: int = 0):
        gen_dl_iter = iter(self.dl_gen)
        for n, batch_disc in enumerate(self.dl_disc):
            # Forward and backward pass for discriminator
            src_imgs: torch.Tensor = batch_disc["source"]\
                    .to(self.device)
            gen_imgs: torch.Tensor = self.generator(src_imgs)
            target_imgs: torch.Tensor = batch_disc["target"]\
                    .to(self.device)
            # Forward and backward pass for discriminator (not gen)
            self._set_disc_trainable(True)
            generated_disc_pair = torch.cat(    # Generated, source
                    [gen_imgs.detach(), src_imgs], dim=1) # No gen
            disc_gen = self.discriminator(generated_disc_pair)  # -> 0
            true_disc_pair = torch.cat(    # Target, source
                    [target_imgs, src_imgs], dim=1)
            disc_true = self.discriminator(true_disc_pair)  # -> 1
            loss_disc_gan = self.loss_desc_gan(disc_gen, disc_true)
            loss_disc_gan.backward()
            # DEBUG: Check gradient histogram
            if n % 10 == 0:     # Gradient norms
                disc_grads_norm = 0.0
                num_params = 0
                for param in self.discriminator.parameters():
                    if param.requires_grad:
                        num_params += 1
                        disc_grads_norm += param.grad.norm().item()
                disc_grads_norm /= num_params
            if n == 5:          # Gradient histograms
                abs_grads_all = []
                for param in self.discriminator.parameters():
                    if param.requires_grad:
                        abs_grads_all.append(torch.abs(param.grad)\
                                    .flatten())
                abs_grads_all = torch.cat(abs_grads_all)
                self.tb_writer.add_histogram("debug/disc_grad_hist",
                        abs_grads_all, curr_epoch)
            self.optim_disc.step()
            self.optim_disc.zero_grad()
            # Forward and backward pass for generator
            self._set_disc_trainable(False) # No grad for disc
            batch_gen = next(gen_dl_iter)   # Different sampler
            src_imgs: torch.Tensor = batch_gen["source"]\
                    .to(self.device)
            gen_imgs: torch.Tensor = self.generator(src_imgs)
            target_imgs: torch.Tensor = batch_gen["target"]\
                    .to(self.device)
            generated_disc_pair_tr = torch.cat( # Grad over gen
                    [gen_imgs, src_imgs], dim=1)
            disc_gen_tr = self.discriminator(
                    generated_disc_pair_tr)
            loss_gen, loss_gen_gan, loss_gen_l1 = self.\
                    loss_gen_gan_l1(gen_imgs, disc_gen_tr, 
                                    target_imgs)
            loss_gen.backward()
            # DEBUG: Check gradient histogram
            if n % 10 == 0:     # Gradient norms
                gen_grads_norm = 0.0
                num_params = 0
                for param in self.generator.parameters():
                    if param.requires_grad:
                        num_params += 1
                        gen_grads_norm += param.grad.norm().item()
                gen_grads_norm /= num_params
            if n == 5:          # Gradient histograms
                abs_grads_all = []
                for param in self.generator.parameters():
                    if param.requires_grad:
                        abs_grads_all.append(torch.abs(param.grad)\
                                    .flatten())
                abs_grads_all = torch.cat(abs_grads_all)
                self.tb_writer.add_histogram("debug/gen_grad_hist",
                        abs_grads_all, curr_epoch)
            self.optim_gen.step()
            self.optim_gen.zero_grad()
            if n % 10 == 0:
                if self.tb_writer is None:
                    print(f"Disc. loss: {loss_disc_gan.item():.3f}, "\
                            f"Gen. loss (GAN + L1 = total): "\
                            f"{loss_gen_gan.item():.3f} "\
                            f"+ {loss_gen_l1.item():.3f} "\
                            f"= {loss_gen.item():.3f}")
                else:
                    gs = curr_epoch * len(self.dl_disc) + n  # Step
                    # self.tb_writer.add_scalars(   # Adds namespaces!
                    #     "loss", {
                    #         "disc": loss_disc_gan.item(),
                    #         "gen_l1": loss_gen_l1.item(),
                    #         "gen_gan": loss_gen_gan.item(),
                    #         "gen_total": loss_gen.item(),
                    #     }, gs)
                    self.tb_writer.add_scalar("loss/disc", 
                            loss_disc_gan.item(), gs)
                    self.tb_writer.add_scalar("loss/gen_l1",
                            loss_gen_l1.item(), gs)
                    self.tb_writer.add_scalar("loss/gen_gan", 
                            loss_gen_gan.item(), gs)
                    self.tb_writer.add_scalar("loss/gen_total",
                            loss_gen.item(), gs)
                    self.tb_writer.add_scalar("debug/disc_grad_norm",
                            disc_grads_norm, gs)
                    self.tb_writer.add_scalar("debug/gen_grad_norm",
                            gen_grads_norm, gs)
                    self.tb_writer.add_scalar("epoch", curr_epoch, gs)
        return loss_disc_gan.item(), \
            (loss_gen_l1.item(), loss_gen_gan.item(), loss_gen.item())
    
    # Get the result from a testing batch (sample) and log it (val)
    def test(self, sample, curr_epoch: int = 0):
        # Generated images from generator
        src_imgs: torch.Tensor = sample["source"].to(self.device)
        tgt_imgs: torch.Tensor = sample["target"].to(self.device)
        with torch.inference_mode():
            gen_imgs: torch.Tensor = self.generator(src_imgs)
        # Stack (along width) and save (or tensorboard)
        fin_imgs = torch.cat(list(map(norm_img, # Norm individually
                [src_imgs, tgt_imgs, gen_imgs])), dim=-1)
        fin_imgs = fin_imgs.cpu().numpy()
        if self.tb_writer is not None:
            self.tb_writer.add_images("val_img", fin_imgs, curr_epoch)
        else:
            sdir = ex(f"./imgs/epoch_{curr_epoch}") # Save directory
            print(f"Saving images to {sdir}")
            if not os.path.exists(sdir):
                os.makedirs(sdir)
            fin_imgs = ein.rearrange(fin_imgs, "b c h w -> b h w c")
            for i, img in enumerate(fin_imgs):
                img = (img * 255).astype(np.uint8)
                Image.fromarray(img).save(f"{sdir}/img_{i}.jpg")
        # Discriminator accuracy
        generated_disc_pair = torch.cat(    # Generated, source
                [gen_imgs, src_imgs], dim=1)
        true_disc_pair = torch.cat(    # Target, source
                [tgt_imgs, src_imgs], dim=1)
        with torch.inference_mode():
            disc_gen = self.discriminator(generated_disc_pair)  # -> 0
            disc_true = self.discriminator(true_disc_pair)  # -> 1
        disc_outputs = torch.cat([disc_gen, disc_true]) # (2*b)
        disc_out_classes = torch.cat(   # 0 (generated), 1 (true)
                [torch.zeros(disc_gen.size()),
                    torch.ones(disc_true.size())])  # (2*b) shape
        disc_out_classes = disc_out_classes.to(self.device)
        disc_outputs = (disc_outputs > disc_outputs.mean()).float()
        disc_accuracy = disc_outputs.eq(disc_out_classes).float()\
                .mean().cpu().item()
        # print(f"{disc_outputs = }\n{disc_out_classes = }")
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("debug/disc_acc", disc_accuracy,
                    curr_epoch)
        else:
            print(f"Discriminator accuracy = {disc_accuracy:.3} %")
    
    def train(self, num_epochs: int = 10):
        self.generator.train()
        self.discriminator.train()
        if self.dl_test is not None:    # Testing when training
            print("Using the test dataset (validation)")
            # assert len(self.dl_test) >= num_epochs, \
            #         f"Insufficient test samples "\
            #         f"({len(self.dl_test)} < {num_epochs})"
            test_sampler = iter(self.dl_test)
            self.test(next(test_sampler), 0)
        else:
            print("No test dataset provided")
        # Main training loop
        for epoch in tqdm(range(num_epochs)):
            disc_loss, gen_loss = self._train_epoch(epoch)
            if self.dl_test is not None:
                try:
                    self.test(next(test_sampler), epoch + 1)
                except StopIteration:   # In case we run out samples!
                    test_sampler = iter(self.dl_test)
                    self.test(next(test_sampler), epoch + 1)
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("epoch/l_disc", disc_loss,
                        epoch + 1)
                self.tb_writer.add_scalar("epoch/l_gen_l1", 
                        gen_loss[0], epoch + 1)
                self.tb_writer.add_scalar("epoch/l_gen_gan", 
                        gen_loss[1], epoch + 1)
                self.tb_writer.add_scalar("epoch/l_gen_total", 
                        gen_loss[2], epoch + 1)
        if self.tb_writer is not None:  # Cleanup tensorboard
            self.tb_writer.flush()
            self.tb_writer.close()

# %%
def verify_args(args: LocalArgs):
    assert args.ds in _data_source.keys(), \
            f"Invalid dataset '{args.ds}', should be in " \
            f"{list(_data_source.keys())}"


if __name__ == "__main__" and "ipykernel" not in sys.argv[0]:
    start_time = time.time()
    args = tyro.cli(LocalArgs)
    verify_args(args)
    print(f"Arguments: {args}")
    # Dataset
    set_download_dir(args.download_dir) # Training set
    ds = Pix2PixDataset(args.ds, swap_st=args.swap_st)
    ds_test = Pix2PixDataset(args.ds, use_split="test", 
            swap_st=args.swap_st)   # Test (validation) set
    dl_test = DataLoader(ds_test, batch_size=5, shuffle=True)
    # Networks
    generator = UNetGenerator()
    discriminator = Discriminator(in_channels=3+3)
    # Trainer module
    use_cuda = args.use_gpu
    if use_cuda:
        assert torch.cuda.is_available(), "CUDA not available"
    trainer = Trainer(ds, generator, discriminator, 
            device="cuda" if use_cuda else "cpu", dl_test=dl_test,
            train_params=args.train)
    trainer.train(args.num_epochs)
    # Save models
    ckpt_dname = os.path.join(args.download_dir, "checkpoints/"\
            f"{datetime.now().strftime(r'%Y-%m-%dT%H-%M-%S')}_"\
            f"{trainer.trainer_id}")    # Timestamp and ID
    if not os.path.isdir(ckpt_dname):
        os.makedirs(ckpt_dname)
    gen_ckpt = f"{ckpt_dname}/res_generator.pt"
    print(f"Saving generator to {gen_ckpt}")
    torch.save(generator.state_dict(), gen_ckpt)
    disc_ckpt = f"{ckpt_dname}/res_discriminator.pt"
    print(f"Saving discriminator to {disc_ckpt}")
    torch.save(discriminator.state_dict(), disc_ckpt)
    param_file = f"{ckpt_dname}/{trainer.trainer_id}_config.yaml"
    print(f"Saving config to {param_file}")
    with open(param_file, "w") as f:
        yaml.safe_dump(asdict(args), f)
    end_time = time.time()
    print("Finished everything")
    print(f"It took {end_time - start_time:.3f} seconds!")
    exit(0)


# %%
# Experimental section
set_download_dir("/scratch/mlhub/pix2pix/")

# %%
# Training test
args = LocalArgs(TrainerArgs(), "cityscapes", swap_st=True,
        download_dir="/scratch/mlhub/pix2pix")
print(f"Arguments: {args}")
verify_args(args)
# Dataset
set_download_dir(args.download_dir) # Training set
ds = Pix2PixDataset(args.ds, swap_st=args.swap_st)
ds_test = Pix2PixDataset(args.ds, use_split="test", 
        swap_st=args.swap_st)   # Test (validation) set
dl_test = DataLoader(ds_test, batch_size=5, shuffle=True)
# Networks
generator = UNetGenerator()
discriminator = Discriminator(in_channels=3+3)
# Trainer module
use_cuda = args.use_gpu
if use_cuda:
    assert torch.cuda.is_available(), "CUDA not available"
trainer = Trainer(ds, generator, discriminator, 
        device="cuda" if use_cuda else "cpu", dl_test=dl_test,
        train_params=args.train)
# %%
trainer.train(args.num_epochs)

# %%
disc_ckpt = torch.load("/scratch/mlhub/pix2pix/checkpoints/2024-05-19T23-02-00/res_discriminator.pt")
disc = Discriminator()
disc.load_state_dict(disc_ckpt)
gen_ckpt = torch.load("/scratch/mlhub/pix2pix/checkpoints/2024-05-19T23-02-00/res_generator.pt")
gen = UNetGenerator()
gen.load_state_dict(gen_ckpt)

# %%
ds_test = Pix2PixDataset("cityscapes", use_split="test", swap_st=True)

# %%
sample = ds_test[200]

# %%
plt.title("Target image")
plt.imshow(ein.rearrange(norm_img(sample["target"]), 
                        "C H W -> H W C").numpy())
plt.show()

# %%
disc(torch.cat([sample["target"], sample["source"]]).unsqueeze(0))

# %%
gen_sample = gen(sample["source"].unsqueeze(0))

# %%
disc(torch.cat([gen_sample[0], sample["source"]]).unsqueeze(0))

# %%
plt.title("Generated image")
plt.imshow(ein.rearrange(norm_img(gen_sample[0]), 
                        "C H W -> H W C").detach().numpy())
plt.show()

# %%
