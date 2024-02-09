
import torch.nn as nn 
import math  
import torch 
from utils.helper import compute_msssim

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=[1e-2], type='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type

    def forward(self, output, target,p = None):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        if p is None:
            p = self.lmbda[0] 

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.type == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = p * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        else:
            out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
            out["loss"] = p * (1 - out['ms_ssim_loss']) + out["bpp_loss"]

        return out