
import argparse
from models import models_dict

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("-m","--model",default= "scale_stanh",choices=models_dict.keys(),help="Model architecture (default: %(default)s)",)
    parser.add_argument("-d", "--dataset", type=str, default = "/scratch/dataset/openimages", help="Training dataset")
    parser.add_argument("-e","--epochs",default=150,type=int,help="Number of epochs (default: %(default)s)", )
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate (default: %(default)s)",)
    parser.add_argument("-n","--num-workers",type=int,default=8,help="Dataloaders threads (default: %(default)s)",)
    parser.add_argument("--num_images",type=int,default=300000,help="Dataloaders threads (default: %(default)s)",)
    parser.add_argument("--num_images_val",type=int,default=816,help="Dataloaders threads (default: %(default)s)",)
    parser.add_argument("--lambda_list",nargs='+', type=float,default= [0.013],help="Bit-rate distortion parameter (default: %(default)s)",)
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: %(default)s)")
    parser.add_argument("--test-batch-size",type=int,default=8,help="Test batch size (default: %(default)s)",)
    parser.add_argument("--aux-learning-rate",default=1e-3,help="Auxiliary loss learning rate (default: %(default)s)",)
    parser.add_argument("--patch-size",type=int,nargs=2,default=(256, 256),help="Size of the patches to be cropped (default: %(default)s)",)
    parser.add_argument("--cuda", action="store_true",  default=True,help="Use cuda")
    parser.add_argument("--save", action="store_true", default=True, help="Save model to disk")
    parser.add_argument("--seed", type=float, default=100, help="Set random seed for reproducibility")
    parser.add_argument("--quality", type=int, default=0, help="quality")
    parser.add_argument("--clip_max_norm",default=1.0,type=float,help="gradient clipping max norm (default: %(default)s",)
    parser.add_argument("--checkpoint", type=str, default = "/scratch/StanhLTM/models/zero_scale_stanh_192_False_25/_very_best.pth.tar",help="Path to a checkpoint") #/scratch/StanhLTM/base_models/0.0067.pth.tar
    parser.add_argument("--type", type=str, default='mse', help="loss type", choices=['mse', "ms-ssim"])
    parser.add_argument("--save_path", type=str, default="/scratch/StanhLTM/models/", help="Where to Save model"
    )
    parser.add_argument( "--skip_epoch", type=int, default=0)
    parser.add_argument("--N", type=int, default=192,)
    parser.add_argument("--M", type=int, default=320,)
    parser.add_argument("--lr_epoch", nargs='+', type=int,default=[50,100,150])
    parser.add_argument("--continue_train", action="store_true") #ddddd
    parser.add_argument("--removing_mean", action="store_true")
    parser.add_argument("--refinement", type = str, default="none",)

    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--fact_stanh", action="store_true")
    parser.add_argument("--tester", action="store_true")
    parser.add_argument("--factorized_stanh", action="store_true")


    parser.add_argument("--gauss_beta",default=[15,15,15],nargs='+', type=int,help="gauss_beta",) #zio caro
    parser.add_argument("--gauss_num_sigmoids",default=[0,0,0],nargs='+', type=int,help="gauss_beta",)
    parser.add_argument("--gauss_extrema",default=[80,80,80],nargs='+', type=int,help="gauss_extrema",)
    parser.add_argument("--gauss_gp",default=[25,25,25],nargs='+', type=int,help="gauss_beta",)
    parser.add_argument("--symmetry",action="store_true",help="factorized_beta",)
    parser.add_argument("--gauss_annealing",default=["gap_stoc","gap_stoc","gap_stoc"],nargs='+', type=str,help="factorized_annealing",)
    parser.add_argument("--gauss_trainable",default=["yes","yes","yes"],nargs='+', type=str,help="gauss_beta")


    parser.add_argument("--wandb_name", type=str, default = "stanh_der",help="Path to a checkpoint")


    args = parser.parse_args(argv)
    return args