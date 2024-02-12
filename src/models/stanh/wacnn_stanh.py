

from ..reference.wacnn import WACNN
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
from entropy_models import GaussianConditionalStanh
from compressai.layers import GDN
from ..reference.utils import deconv

import numpy as np
from ..reference.layers import Win_noShift_Attention
import math


SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x

def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)

def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)

def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')

def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    if not module:
        return
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}", 
            state_dict,
            policy,
            dtype,
        )

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

class WACNN_stanh(WACNN):
    """CNN based model"""

    def __init__(self, 
                 gaussian_configuration, 
                 lambda_list, 
                 N=192, 
                 M=320,
                 refinement = "none",
                 **kwargs):
        super().__init__(N = N, M = M ,**kwargs)


        assert refinement in ["none","convolution","multiple"]
        self.refinement = refinement
        self.lmbda = lambda_list
        self.levels = len(self.lmbda)

        self.gaussian_configuration = gaussian_configuration
        self.gaussian_conditional = nn.ModuleList(GaussianConditionalStanh(None,
                                                    channels = N,
                                                    gaussian_configuration =self.gaussian_configuration[i],
                                                            )
                                                for i in range(self.levels)
                                                )
        

        if self.refinement == "multiple":
            self.g_s = nn.ModuleList(nn.Sequential(
                Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
                deconv(M, N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                deconv(N, N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
                deconv(N, N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                deconv(N, 3, kernel_size=5, stride=2),
            )
            for _ in range(self.levels)
            )
        elif self.refinement == "convolution":
            self.refine_layer = nn.ModuleList(nn.Conv2d(in_channels=M,
                                                       out_channels=M,
                                                         kernel_size=1, 
                                                         stride=1, 
                                                         padding=0)
                                            for _ in range(self.levels))
        else: 
            self.g_s = nn.Sequential(
                Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
                deconv(M, N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                deconv(N, N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
                deconv(N, N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                deconv(N, 3, kernel_size=5, stride=2),
            )          


    def print_information(self):
        print(" g_a: ",sum(p.numel() for p in self.g_a.parameters()))

        if self.refinement == "multiple":
            for i in range(self.levels):
                print(" g_s " + str(i) + " :",sum(p.numel() for p in self.g_s[i].parameters()))
        elif self.refinement == "convolution":
            for i in range(self.levels):
                print("refinement " + str(i),": ",sum(p.numel() for p in self.refine_layer[i].parameters()))

        print(" h_a: ",sum(p.numel() for p in self.h_a.parameters()))


        print(" h_means_a: ",sum(p.numel() for p in self.h_mean_s.parameters()))
        print(" h_scale_a: ",sum(p.numel() for p in self.h_scale_s.parameters()))

        print("cc_mean_transforms",sum(p.numel() for p in self.cc_mean_transforms.parameters()))
        print("cc_scale_transforms",sum(p.numel() for p in self.cc_scale_transforms.parameters()))


        print("lrp_transform",sum(p.numel() for p in self.lrp_transforms.parameters()))


        print(" TRAINABLE STANH",sum(p.numel() for p in self.gaussian_conditional[0].stanh.parameters() if p.requires_grad))
        print(" FROZEN STANH",sum(p.numel() for p in self.gaussian_conditional[0].stanh.parameters() if p.requires_grad == False))

        print("**************************************************************************")
        model_tr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_fr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad== False)
        print(" trainable parameters: ",model_tr_parameters)
        print(" freeze parameters: ", model_fr_parameters)
        return sum(p.numel() for p in self.parameters() if p.requires_grad) #dddd




    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        for i in range(self.levels):
            updated = self.gaussian_conditional[i].update_scale_table(scale_table)
        self.entropy_bottleneck.update(force = force)
        return updated


    def define_permutation(self, x):
        perm = np.arange(len(x.shape)) 
        perm[0], perm[1] = perm[1], perm[0]
        inv_perm = np.arange(len(x.shape))[np.argsort(perm)] # perm and inv perm
        return perm, inv_perm 
    

    def compute_gap(self, inputs, y_hat,lv):
        perm, _ = self.define_permutation(inputs)
        values =  inputs.permute(*perm).contiguous() # flatten y and call it values
        values = values.reshape(1, 1, -1) # reshape values      
        y_hat_p =  y_hat.permute(*perm).contiguous() # flatten y and call it values
        y_hat_p = y_hat_p.reshape(1, 1, -1) # reshape values     
        with torch.no_grad():    
            out = self.gaussian_conditional[lv].stanh(values,-1) 
            # calculate f_tilde:  
            f_tilde = F.mse_loss(values, y_hat_p)
            # calculat f_hat
            f_hat = F.mse_loss(values, out)
            gap = torch.abs(f_tilde - f_hat)
        return gap

    def forward(self, x, tr = True, lv = 0):
        self.gaussian_conditional[lv].stanh.update_state(x.device)
        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):

            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            y_hat_slice, y_slice_likelihood = self.gaussian_conditional[lv](y_slice, scale, means = mu,training = tr)
            #_, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            #y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)

        y_gap = self.gaussian_conditional[lv].quantize(y, mode ="training")
        gap_gaussian = self.compute_gap(y,  y_gap, lv)

        y_likelihoods = torch.cat(y_likelihood, dim=1)
        if self.refinement == "multiple":
            x_hat = self.g_s[lv](y_hat)
        elif self.refinement == "convolution":
            y_hat = self.refine_layer[lv](y_hat)
            x_hat = self.g_s(y_hat)
        else:
            x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "gap_gaussian":gap_gaussian
        }