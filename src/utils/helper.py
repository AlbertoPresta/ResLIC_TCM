
from pytorch_msssim import ms_ssim
import torch.nn as nn
from annealings import StHAnnealing, Annealing_triangle, RandomAnnealings
import torch 
import math

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def configure_latent_space_policy(args):

    gaussian_configuration = {
                "beta": args.gauss_beta, 
                "num_sigmoids": args.gauss_num_sigmoids, 
                "gauss_annealing": args.gauss_annealing, 
                "symmetry": args.symmetry, 
                "gap_factor": args.gauss_gp ,
                "extrema": args.gauss_extrema ,
            "trainable":args.gauss_tr         
            }

    return  gaussian_configuration



from datetime import datetime
from os.path import join


def create_savepath(args,epoch,base_path):
    now = datetime.now()
    date_time = now.strftime("%m%d")
    c = join(date_time,"_lambda_",str(args.lmbda_starter),"_epoch_",str(epoch)).replace("/","_")

    
    c_best = join(c,"best").replace("/","_")
    c = join(c,".pth.tar").replace("/","_")
    c_best = join(c_best,".pth.tar").replace("/","_")
    
    
    
    savepath = join(base_path,c)
    savepath_best = join(base_path,c_best)
    
    print("savepath: ",savepath)
    print("savepath best: ",savepath_best)
    return savepath, savepath_best



def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)



class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)
        



def configure_annealings(gaussian_configuration):

  
    if "random" in gaussian_configuration["annealing"]:
        annealing_strategy_gaussian = RandomAnnealings(beta = gaussian_configuration["beta"],  type = gaussian_configuration["annealing"], gap = False)
  
    elif "triangle" in gaussian_configuration["annealing"]:
        annealing_strategy_gaussian = Annealing_triangle(beta = gaussian_configuration["beta"], factor = gaussian_configuration["gap_factor"])
    
    else:
        annealing_strategy_gaussian = StHAnnealing(beta = gaussian_configuration["beta"], 
                                    factor = gaussian_configuration["gap_factor"], 
                                    type = gaussian_configuration["annealing"]) 
    

    return  annealing_strategy_gaussian
