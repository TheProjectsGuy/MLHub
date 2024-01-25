# %%
import sys
import time
import torch
import numpy as np
import einops as ein
import torch.nn as nn
from torchinfo import summary
from torch.nn import functional as F
from typing import Optional, Union, List


# %%
class SigmoidSquashingActivation(nn.Module):
    """
        Sigmoid squashing activation function from the LeNet paper. It
        is a scaled hyperbolic tangent function.
        Function does f(x) = A tanh(S*x)
    """
    def __init__(self, A = 1.7159, S = 2/3) -> None:
        """
            - A: scaling for output of tanh
            - S: scaling for input
        """
        super().__init__()
        self.A = A
        self.S = S
    
    def forward(self, x):
        return self.A * torch.tanh(self.S * x)


# %%
class SubSamplingLayer(nn.Module):
    """
        Sub-sampling layer for LeNet5.
    """
    def __init__(self, in_channels: int, kernel_size: int = 2) \
            -> None:
        """
            - in_channels: Number of channels in the input image. The
                output has the same number of channels.
            - kernel_size: The size of the kernel to use for 
                sub-sampling
        """
        super().__init__()
        out_channels = in_channels
        self.kernel_size = kernel_size
        self.weights = nn.Parameter(
                torch.randn([1, in_channels, 1, 1]))
        self.kernel = nn.Parameter(torch.ones(
            [out_channels, in_channels, kernel_size, kernel_size]) \
                    * 0.25 * self.weights)
        self.bias = nn.Parameter(torch.randn([out_channels]))
    
    def forward(self, x):
        k = self.kernel.to(x.device)
        return F.conv2d(x, k, self.bias, stride=2)


# %%
class CustomConvLayer(nn.Module):
    """
        Custom convolution layer for LeNet5. Details in Table 1 and
        related text in the paper [1].
        
        1: LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.
    """
    def __init__(self, in_channels: int = 6, out_channels: int = 16, 
                kernel_size: int = 5, bias: bool = True,
                connected_map: Union[np.ndarray, str] \
                    = "default") -> None:
        """
            - in_channels: Number of channels in the input image.
            - out_channels: Number of channels in the output image.
            - kernel_size: The size of the kernel to use for 
                convolution.
            - bias: If True, use bias. Else bias is 0.
            - connected_map: The connectivity map to use for 
                convolution. If "default", the connectivity map is
                taken from the table 1 of the paper [1]. If a numpy
                array is given, it should be a boolean array of shape 
                [out_channels, in_channels] where [i, j] is the ith
                out connected to jth in channel (bool).
        """
        super().__init__()
        if connected_map == "default":
            assert in_channels == 6 and out_channels == 16, \
                "Default map requires in and out channels to be 6 "\
                    "and 16, respectively "\
                    f"({in_channels = }, {out_channels = })"
            connected_map = np.array([   # [in_ch, ...] # out-ch
                [True , True , True , False, False, False], # 0
                [False, True , True , True , False, False], # 1
                [False, False, True , True , True , False], # 2
                [False, False, False, True , True , True ], # 3
                [True , False, False, False, True , True ], # 4
                [True , True , False, False, False, True ], # 5
                [True , True , True , True , False, False], # 6
                [False, True , True , True , True , False], # 7
                [False, False, True , True , True , True ], # 8
                [True , False, False, True , True , True ], # 9
                [True , True , False, False, True , True ], # 10
                [True , True , True , False, False, True ], # 11
                [True , True , False, True , True , False], # 12
                [False, True , True , False, True , True ], # 13
                [True , False, True , True , False, True ], # 14
                [True , True , True , True , True , True ], # 15
            ])
        assert isinstance(connected_map, np.ndarray) \
            and connected_map.dtype == bool \
            and connected_map.shape == (out_channels, in_channels)
        # Trainable parameters
        self.tr_params: List[nn.Parameter] = []
        self.ks = kernel_size
        self.connected_map = connected_map
        self.out_channels = out_channels
        self.in_channels = in_channels
        ks = self.ks
        for i, ch_conn in enumerate(self.connected_map):
            tr_param = nn.Parameter(
                    torch.randn([1, ch_conn.sum(), ks, ks]))
            self.register_parameter(f"w_ol_{i}", tr_param)
            self.tr_params.append(tr_param)
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.randn([out_channels]))
        else:
            self.bias = None
    
    def _create_weight(self):
        """
            Create the weight tensor from the trainable parameters.
        """
        ks = self.ks
        out_c, in_c = self.out_channels, self.in_channels
        dev = self.tr_params[0].device
        weight = torch.zeros([out_c, in_c, ks, ks], device=dev)
        for i, ch_conn in enumerate(self.connected_map):
            weight[i, ch_conn] = self.tr_params[i]
        return weight
    
    def forward(self, x):
        weight = self._create_weight()
        return F.conv2d(x, weight, self.bias)


# %%
class RBFUnits(nn.Module):
    """
        Radial Basis Function units for the final classification head
        of LeNet.
    """
    def __init__(self, in_features: int = 84, out_features: int = 10,
                param_vect: Union[np.ndarray, str] = "default",
                requires_grad: bool = False) \
                -> None:
        """
            - in_features: Number of input features.
            - out_features: Number of output features.
            - param_vect: Parameter vector for the RBF units. Should
                be a numpy array. If "default", then the default
                from Figure 3 is used.
            - requires_grad: If True, the gradient is enabled, else
                no gradient is enabled (no backprop over parameters)
        """
        super().__init__()
        self.in_d = in_features
        self.out_d = out_features
        if param_vect == "default":
            param_vect = np.array([ # List of each unit
                [   # 0; Each unit is a (12, 7) character repr
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0],
                    [0, 1, 1, 0, 1, 1, 0],
                    [1, 1, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 1, 1],
                    [0, 1, 1, 0, 1, 1, 0],
                    [0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ], [    # 1
                    [0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ], [    # 2
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [1, 1, 0, 0, 0, 1, 1],
                    [1, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 1, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ], [    # 3
                    [1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1],
                    [1, 0, 0, 0, 0, 1, 1],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ], [    # 4
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 1, 1],
                    [0, 1, 1, 0, 0, 1, 1],
                    [1, 1, 1, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1],
                ], [    # 5
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 1, 1],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ], [    # 6
                    [0, 0, 1, 1, 1, 1, 0],
                    [0, 1, 1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 1, 1],
                    [1, 1, 1, 0, 0, 1, 1],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ], [    # 7
                    [1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ], [    # 8
                    [0, 1, 1, 1, 1, 1, 0],
                    [1, 1, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 1, 1],
                    [0, 1, 1, 1, 1, 1, 0],
                    [1, 1, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 1, 1],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ], [    # 9
                    [0, 1, 1, 1, 1, 1, 0],
                    [1, 1, 0, 0, 1, 1, 1],
                    [1, 1, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1, 0],
                    [0, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ], dtype=np.float32)
            param_vect = param_vect * 2 - 1 # -1 and +1 range
            param_vect = ein.rearrange(param_vect, "l h w -> l (h w)")
        assert isinstance(param_vect, np.ndarray) \
            and param_vect.shape == (self.out_d, self.in_d) \
            and param_vect.dtype == np.float32
        self.param_vect = nn.Parameter(torch.from_numpy(param_vect), 
                requires_grad=requires_grad)
    
    def forward(self, x):
        ret_batching = True
        if len(x.shape) == 1:   # Add b = 1 and remove it in the end
            ret_batching = False
            x = ein.rearrange(x, "i -> 1 i")
        x = ein.rearrange(x, "b i_c -> b 1 i_c")
        w = ein.rearrange(self.param_vect, "o i_c -> 1 o i_c")
        res = torch.sum((x - w) ** 2, dim=2)
        if not ret_batching:    # No batching used for input
            res = res[0]
        return res


# %%
class LeNet5(nn.Module):
    """
        LeNet5 network presented in section 2 of the paper [1].
        
        1: LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.
    """
    def __init__(self) -> None:
        super().__init__()
        # C1: convolution layer
        self.c1 = nn.Conv2d(1, 6, 5)
        self.c1a = SigmoidSquashingActivation()
        # S2: sub-sampling layer
        self.s2 = SubSamplingLayer(6)
        self.s2a = SigmoidSquashingActivation()
        # C3: custom convolution layer
        self.c3 = CustomConvLayer(6, 16, 5)
        self.c3a = SigmoidSquashingActivation()
        # S4: sub-sampling layer
        self.s4 = SubSamplingLayer(16)
        self.s4a = SigmoidSquashingActivation()
        # C5: convolution layer
        self.c5 = nn.Conv2d(16, 120, 5)
        self.c5a = SigmoidSquashingActivation()
        # F6: fully connected layer
        self.f6 = nn.Linear(120, 84)
        self.f6a = SigmoidSquashingActivation()
        # Out: output layer
        self.rbf_out = RBFUnits(84, 10)
    
    def forward(self, x):
        x = self.c1a(self.c1(x))
        x = self.s2a(self.s2(x))
        x = self.c3a(self.c3(x))
        x = self.s4a(self.s4(x))
        x = self.c5a(self.c5(x))
        x = x.squeeze() # ([b], c=120, 1, 1) -> ([b], c)
        x = self.f6a(self.f6(x))
        x = self.rbf_out(x)
        return x


# %%
if __name__ == "__main__" and "ipykernel" not in sys.argv[0]:
    print("Testing the LeNet-5 network (forward pass and FLOPS)")
    use_gpu = torch.cuda.is_available()
    model = LeNet5()
    sample_in = torch.randn(16, 1, 32, 32)
    if use_gpu:
        print("GPU found")
        model.cuda()
        sample_in = sample_in.cuda()
    else:
        print("GPU not found, running on CPU")
    print(f"Model summary: {model}")
    summary(model, sample_in.shape)
    start_time = time.time()
    sample_out = model(sample_in)
    end_time = time.time()
    print(f"Model input shape: {sample_in.shape}")
    print(f"Model output shape: {sample_out.shape}")
    # Per sample statistic
    dur = (end_time - start_time)/sample_in.shape[0]
    freq_hz = 1/dur
    print(f"Time: {dur:.3f} secs ({freq_hz:.3f} Hz)")


# %%
