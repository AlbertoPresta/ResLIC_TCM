
from pytorch_msssim import ms_ssim
import torch.nn as nn
from annealings import StanhAnnealings, Annealing_triangle, RandomAnnealings
import torch 
import math
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def configure_latent_space_policy(args, multi = False):
    if multi is False:
        gauss_tr = True if args.gauss_trainable[0] == "yes" else False
        gaussian_configuration = {
                    "beta": args.gauss_beta[0] if isinstance(args.gauss_beta,list) else args.gauss_beta, 
                    "num_sigmoids": args.gauss_num_sigmoids[0] if isinstance(args.gauss_num_sigmoids,list) else args.gauss_num_sigmoids, 
                    "annealing": args.gauss_annealing[0] if isinstance(args.gauss_annealing,list) else args.gauss_annealing, 
                    "symmetry": args.symmetry, 
                    "gap_factor": args.gauss_gp[0] if isinstance(args.gauss_gp,list) else args.gauss_gp,
                    "extrema": args.gauss_extrema[0] if isinstance(args.gauss_extrema,list) else args.gauss_extrema ,
                "trainable":gauss_tr ,
                "removing_mean":args.removing_mean  #dddd      
                }
        return  gaussian_configuration #[gaussian_configuration]
    else: 
        gaussian_configuration = []
        for i in range(len(args.lambda_list)):
            gauss_tr = True if args.gauss_trainable[i] == "yes" else False
            gaussian_configuration.append({
                        "beta": args.gauss_beta[i], 
                        "num_sigmoids": args.gauss_num_sigmoids[i], 
                        "annealing": args.gauss_annealing[i], 
                        "symmetry": args.symmetry, 
                        "gap_factor": args.gauss_gp[i] ,
                        "extrema": args.gauss_extrema[i] ,
                    "trainable":gauss_tr ,
                    "removing_mean":args.removing_mean #dddd      
                    }
                )
        return  gaussian_configuration




from datetime import datetime
from os.path import join


def create_savepath(args,epoch,base_path):
    now = datetime.now()
    date_time = now.strftime("%m%d")
    c = join(date_time,"_lambda_",str(args.lambda_list[0]),"_epoch_",str(epoch)).replace("/","_")

    
    c_best = join(c,"best").replace("/","_")
    c = join(c,".pth.tar").replace("/","_")
    c_best = join(c_best,".pth.tar").replace("/","_")
    
    
    
    savepath = join(base_path,c)
    savepath_best = join(base_path,c_best)
    
    print("savepath: ",savepath)
    print("savepath best: ",savepath_best)
    very_best  = join(base_path,"_very_best.pth.tar")
    return savepath, savepath_best,very_best


def read_image(filepath, ):
    #assert filepath.is_file()
    img = Image.open(filepath)
    img = img.convert("RGB")
    return transforms.ToTensor()(img)


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
        annealing_strategy_gaussian = StanhAnnealings(beta = gaussian_configuration["beta"], 
                                    factor = gaussian_configuration["gap_factor"], 
                                    type = gaussian_configuration["annealing"]) 
    

    return  annealing_strategy_gaussian
