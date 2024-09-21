# File containing models for Pix2Pix work
"""
"""

# %%
import torch
import numpy as np
import einops as ein
from torch import nn
from typing import Tuple
from typing import Optional
from torchinfo import summary
import torch.nn.functional as F


# %%
class EncDecGenerator(nn.Module):
    """
        Encoder-Decoder architecture for Pix2Pix
    """
    def __init__(self, enc_spec: str = "C64-C128-C256-C512-C512-"\
            "C512-C512-C512", dec_spec: str = "CD512-CD512-CD512-"\
            "CD512-C256-C128-C64", in_channels: int = 3, 
            out_channels: int = 3, btlnk_dim: int = 512):
        """
            - enc_spec: Encoder specification
            - dec_spec: Decoder specification
            - in_channels: Number of input channels
            - out_channels: Number of output channels
            - btlnk_dim:   Dimension of the bottleneck layer
        """
        super().__init__()
        enc_layers, _ = self.generate_conv_layers(enc_spec, 
                in_dim=in_channels, ctype="down")
        self.enc_layers = nn.ModuleList(enc_layers)
        dec_layers, nc = self.generate_conv_layers(dec_spec, 
                in_dim=btlnk_dim, ctype="up")
        self.dec_layers = nn.ModuleList(dec_layers)
        # Conv layer to map it out
        self.out_conv = nn.ConvTranspose2d(nc, out_channels, 
                kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.out_act = nn.Tanh()
    
    @staticmethod
    def generate_conv_layers(spec: str, in_dim: int = 3, 
            ctype: str="down", kernel_size: tuple[int, int] = (4, 4),
            stride: tuple[int, int] = (2, 2), padding: int = 1, 
            l1bn: bool = False) -> tuple[nn.Module, int]:
        """
            - spec: String specification. Ck is Conv-BatchNorm-ReLU 
                    with k number of filters. CDk is Conv-BatchNorm-
                    Dropout-ReLU with k number of filters.
                    Eg: "CD512-CD512-CD512-C512-C256-C128-C64"
            - in_dim:   Number of input channels
            - ctype:    Convolution type: should be "up" (for 
                        transpose) or "down" (for regular)
            - kernel_size: Kernel size
            - stride:    Stride
            - padding:   Padding
            - l1bn:     Apply BatchNorm to the first layer (if True)
            
            Returns:
            - all_layers: nn.Sequential     A list of all layers
            - c_dim:    int                 Number of output channels
        """
        all_layers = []
        c_dim = in_dim
        layer_specs = spec.split("-")
        for l, ls in enumerate(layer_specs):
            layers = nn.Sequential()    # Cascade the sub-layers
            if ls.startswith("CD"):
                nk = int(ls[2:])
                use_dropout = True
            elif ls.startswith("C"):
                nk = int(ls[1:])
                use_dropout = False
            else:
                raise ValueError(f"Invalid layer spec '{ls}'")
            # Convolution layer
            if ctype == "down":
                layers.append(nn.Conv2d(c_dim, nk, 
                        kernel_size=kernel_size, stride=stride, 
                        padding=padding))
            elif ctype == "up":
                layers.append(nn.ConvTranspose2d(c_dim, nk,
                        kernel_size=kernel_size, stride=stride,
                        padding=padding))
            else:
                raise ValueError(f"Invalid conv type '{ctype}'")
            # BatchNorm layer
            if l == 0 and l1bn:
                layers.append(nn.BatchNorm2d(nk))
            # Dropout layer
            if use_dropout:
                layers.append(nn.Dropout(0.5))
            # ReLU layer
            if ctype == "down":
                layers.append(nn.LeakyReLU(0.2))
            elif ctype == "up":
                layers.append(nn.ReLU())
            all_layers.append(layers)   # Add the sub-layers
            c_dim = nk      # Input channels for the next layer
        return all_layers, c_dim
    
    def forward(self, x):
        for l in self.enc_layers:
            x = l(x)
        for l in self.dec_layers:
            x = l(x)
        # Final layer
        x = self.out_conv(x)
        x = self.out_act(x)
        return x


# %%
class UNetGenerator(nn.Module):
    """
        UNet architecture for Pix2Pix. Has skip connections 
        (concatenation) from layer 'i' in encoder to layer 'n-i' in
        decoder.
    """
    def __init__(self, enc_spec: str = "C64-C128-C256-C512-C512-"\
            "C512-C512-C512", dec_spec: str = "CD512:512-CD1024:512-"\
            "CD1024:512-C1024:512-C1024:256-C512:128-C256:64-C128:3",
            in_channels: int = 3) -> None:
        """
            - enc_spec: Encoder specification
            - dec_spec: Decoder specification. Note that channels are
                of the form in:out, instead of a single number.
            - in_channels: Number of input channels
            - out_channels: Number of output channels
            - btlnk_dim:   Dimension of the bottleneck layer
        """
        super().__init__()
        enc_layers, _ = self.generate_enc_conv_layers(enc_spec, 
                in_dim=in_channels)
        self.enc_layers = nn.ModuleList(enc_layers)
        dec_layers, _ = self.generate_dec_conv_layers(dec_spec)
        self.dec_layers = nn.ModuleList(dec_layers)
        self.out_act = nn.Tanh()
    
    @staticmethod
    def generate_enc_conv_layers(spec: str, in_dim: int = 3, 
            kernel_size: tuple[int, int] = (4, 4), 
            stride: tuple[int, int] = (2, 2), padding: int = 1, 
            l1bn: bool = False) -> tuple[nn.Module, int]:
        """
            Generate the encoder convolutional layers
            
            - spec: String specification. Ck is Conv-BatchNorm-ReLU 
                    with k number of filters. CDk is Conv-BatchNorm-
                    Dropout-ReLU with k number of filters.
                    Eg: "CD512-CD512-CD512-C512-C256-C128-C64"
            - in_dim:   Number of input channels
            - kernel_size: Kernel size
            - stride:    Stride
            - padding:   Padding
            - l1bn:     Apply BatchNorm to the first layer (if True)
        """
        all_layers = []
        c_dim = in_dim
        layer_specs = spec.split("-")
        for l, ls in enumerate(layer_specs):
            layers = nn.Sequential()    # Cascade the sub-layers
            if ls.startswith("CD"):
                nk = int(ls[2:])
                use_dropout = True
            elif ls.startswith("C"):
                nk = int(ls[1:])
                use_dropout = False
            else:
                raise ValueError(f"Invalid layer spec '{ls}'")
            layers.append(nn.Conv2d(c_dim, nk, # Convolution layer
                    kernel_size=kernel_size, stride=stride, 
                    padding=padding))
            if l == 0 and l1bn: # Apply Batch Norm to first layer
                layers.append(nn.BatchNorm2d(nk))
            if use_dropout:
                layers.append(nn.Dropout(0.5))
            # LeakyReLU Activation for encoder
            layers.append(nn.LeakyReLU(0.2))
            all_layers.append(layers)   # Add the sub-layers
            c_dim = nk      # Input channels for the next layer
        return all_layers, c_dim
    
    @staticmethod
    def generate_dec_conv_layers(spec: str, 
            kernel_size: tuple[int, int] = (4, 4),
            stride: tuple[int, int] = (2, 2), padding: int = 1,
            l1bn: bool = False) -> tuple[nn.Module, int]:
        """
            Generate the decoder convolutional layers
            
            - spec: String specification. Ck is Conv-BatchNorm-ReLU
                    with k number of filters. CDk is Conv-BatchNorm-
                    Dropout-ReLU with k number of filters. Should have
                    input:output channels per block.
                    Eg: "CD512:512-CD1024:512-C1024:512-C1024:3"
            - kernel_size: Kernel size
            - stride:    Stride
            - padding:   Padding
            - l1bn:     Apply BatchNorm to the first layer (if True)
        """
        all_layers = []
        layer_specs = spec.split("-")
        for l, ls in enumerate(layer_specs):
            layers = nn.Sequential()    # Cascade the sub-layers
            if ls.startswith("CD"):
                io_channels = map(int, ls[2:].split(":"))
                use_dropout = True
            elif ls.startswith("C"):
                io_channels = map(int, ls[1:].split(":"))
                use_dropout = False
            else:
                raise ValueError(f"Invalid layer spec '{ls}'")
            in_ch, out_ch = io_channels
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, 
                    kernel_size=kernel_size, stride=stride,
                    padding=padding))
            if l == 0 and l1bn: # Apply Batch Norm to first layer
                layers.append(nn.BatchNorm2d(out_ch))
            if use_dropout:
                layers.append(nn.Dropout(0.5))
            # Up-conv in decoder have ReLU activation
            layers.append(nn.ReLU())
            all_layers.append(layers)   # Add the sub-layers
        return all_layers, out_ch
    
    def forward(self, x):
        enc_outputs = []
        for l in self.enc_layers:
            x = l(x)
            enc_outputs.append(x)
        x = enc_outputs.pop()   # Last output
        for l in self.dec_layers:
            x = l(x)
            if len(enc_outputs) > 0:    # Until residuals remain
                x = torch.cat([x, enc_outputs.pop()], dim=-3)
        # Final layer
        x = self.out_act(x)
        return x


# %%
class Discriminator(nn.Module):
    """
        Discriminator network for Pix2Pix
    """
    def __init__(self, spec: str = "C64-C128-C256-C512",
            kernel_size: Optional[tuple[int, int]] = None, 
            stride: Optional[tuple[int, int]] = None,
            padding: Optional[int] = None,
            in_channels: int = 6) -> None:
        """
            - spec: String specification. Ck is Conv-BatchNorm-ReLU
                    with k number of filters. Also see note 1.
                    Eg: "C64-C128-C256-C512"
            - kernel_size:  Size of the kernel for conv layers. It is
                            (4, 4) if None.
            - stride:       Stride for conv layers. If None, then the
                            stride is (2, 2) for all conv layers but
                            the last one, and (1, 1) for the last one.
            - padding:      Padding for conv layers. It is 1 if None.
            - in_channels:  Number of input channels
            
            Note 1:
            The BatchNorm layer is not applied to the first
            convolution block. All activations are LeakyReLU with
            slope 0.2 (including last layer). After all the conv
            blocks in the "spec", the output goes through a final
            reducing conv layer that maps output to single channel.
            The stride of this layer is (1, 1), the kernel size and
            padding are from the arguments. The output is passed
            through sigmoid, flattened, and averaged.
            
            The result form the above convolutions is averaged (across
            all values in height and width) in the batch and this is
            considered as the final discriminator output.
        """
        super().__init__()
        if kernel_size is None:
            kernel_size = (4, 4)
        if stride is None:
            stride = (2, 2)
            last_stride_one = True
        else:
            last_stride_one = False
        if padding is None:
            padding = 1
        backbone, out_channels = self.generate_conv_layers(spec, 
                in_channels, kernel_size=kernel_size, stride=stride, 
                padding=padding, last_stride_one=last_stride_one)
        self.backbone = nn.ModuleList(backbone)
        self.last_conv = nn.Conv2d(out_channels, 1, 
                kernel_size=kernel_size, stride=(1, 1), 
                padding=padding)
        self.act = nn.Sigmoid()
    
    @staticmethod
    def generate_conv_layers(spec: str, in_dim: int,
            kernel_size: tuple[int, int] = (4, 4),
            stride: tuple[int, int] = (2, 2), padding: int = 1, 
            last_stride_one: bool = True) -> nn.Module:
        """
            Generate the convolutional layers for the given 
            specification. The first conv layer has no BatchNorm.
            
            - spec: String specification. Ck is Conv-BatchNorm-ReLU
                    with k number of filters.
                    Eg: "C64-C128-C256-C512"
            - in_dim:       Number of input channels
            - kernel_size:  Size of the kernel for conv layers.
            - stride:       Stride for conv layers.
            - padding:      Padding for conv layers.
            - last_stride_one:  If True, the last conv layer has
                                stride (1, 1) instead of the given
                                value. If False, then the stride stays
                                the same as the given value.
        """
        all_layers = []
        layer_specs = spec.split("-")
        c_dim = in_dim
        for l, ls in enumerate(layer_specs):
            layers = nn.Sequential()    # Cascade the sub-layers
            if ls.startswith("C"):
                nk = int(ls[1:])    # Number of output channels
            else:
                raise ValueError(f"Invalid layer spec '{ls}'")
            # Conv layer
            if l == len(layer_specs) - 1 and last_stride_one:
                layers.append(nn.Conv2d(c_dim, nk, 
                        kernel_size=kernel_size, stride=(1, 1), 
                        padding=padding))
            else:
                layers.append(nn.Conv2d(c_dim, nk,
                        kernel_size=kernel_size, stride=stride,
                        padding=padding))
            # BatchNorm
            if l > 0:
                layers.append(nn.BatchNorm2d(nk))
            # ReLU
            layers.append(nn.LeakyReLU(0.2))
            all_layers.append(layers)
            c_dim = nk      # Input channels for the next layer
        return all_layers, c_dim    # Layers and num. of out channels
    
    def forward(self, x: torch.Tensor):
        for l in self.backbone:
            x = l(x)
        x = self.last_conv(x)
        x = self.act(x)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Batch it
        x = ein.rearrange(x, "b 1 h w -> b (h w)")
        x = torch.mean(x, dim=1)    # [b] shape
        return x


# %%
if __name__ == "__main__":
    print(f"{':'*20} Encoder Decoder Model {':'*20}")
    test_img = torch.randn(8, 3, 256, 256)
    enc_dec = EncDecGenerator()
    res: torch.Tensor = enc_dec(test_img)
    print(f"Encoder-Decoder result shape: {res.shape}")
    summary(enc_dec, test_img.shape)
    print()
    
    print(f"{':'*20} UNet Generator Model {':'*20}")
    unet_gen = UNetGenerator()
    res: torch.Tensor = unet_gen(test_img)
    print(f"UNet result shape: {res.shape}")
    summary(unet_gen, test_img.shape)
    print()
    
    print(f"{':'*20} Discriminator Model {':'*20}")
    test_imgs = torch.randn(8, 6, 256, 256)
    disc = Discriminator(in_channels=6)
    res: torch.Tensor = disc(test_imgs)
    print(f"Discriminator result shape: {res.shape}")
    summary(disc, test_imgs.shape)
    print()

# %%
# Experimental section
