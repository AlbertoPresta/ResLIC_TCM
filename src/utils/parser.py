
import argparse
from models import models_dict

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("-m","--model",default="tcm_stanh",choices=models_dict.keys(),help="Model architecture (default: %(default)s)",)
    parser.add_argument("-d", "--dataset", type=str, default = "/scratch/dataset/openimages", help="Training dataset")
    parser.add_argument("-e","--epochs",default=250,type=int,help="Number of epochs (default: %(default)s)", )
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate (default: %(default)s)",)
    parser.add_argument("-n","--num-workers",type=int,default=8,help="Dataloaders threads (default: %(default)s)",)
    parser.add_argument("--lambda",dest="lmbda",type=float,default=0.0067,help="Bit-rate distortion parameter (default: %(default)s)",)
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: %(default)s)")
    parser.add_argument("--test-batch-size",type=int,default=8,help="Test batch size (default: %(default)s)",)
    parser.add_argument("--aux-learning-rate",default=1e-3,help="Auxiliary loss learning rate (default: %(default)s)",)
    parser.add_argument("--patch-size",type=int,nargs=2,default=(256, 256),help="Size of the patches to be cropped (default: %(default)s)",)
    parser.add_argument("--cuda", action="store_true",  default=True,help="Use cuda")
    parser.add_argument("--save", action="store_true", default=True, help="Save model to disk")
    parser.add_argument("--seed", type=float, default=100, help="Set random seed for reproducibility")
    parser.add_argument("--clip_max_norm",default=1.0,type=float,help="gradient clipping max norm (default: %(default)s",)
    parser.add_argument("--checkpoint", type=str, default = "none",help="Path to a checkpoint")
    parser.add_argument("--type", type=str, default='mse', help="loss type", choices=['mse', "ms-ssim"])
    parser.add_argument("--save_path", type=str, default="/scratch/StanhLTM/models/", help="Where to Save model"
    )
    parser.add_argument( "--skip_epoch", type=int, default=0)
    parser.add_argument("--N", type=int, default=128,)
    parser.add_argument("--lr_epoch", nargs='+', type=int)
    parser.add_argument("--continue_train", action="store_true", default=True)


    parser.add_argument("--gauss_beta",default=10,type=float,help="gauss_beta",)
    parser.add_argument("--gauss_num_sigmoids",default=0,type=int,help="gauss_beta",)
    parser.add_argument("--gauss_extrema",default=60,type=int,help="gauss_extrema",)
    parser.add_argument("--gauss_gp",default=15,type=int,help="gauss_beta",)
    parser.add_argument("--symmetry",action="store_true", default=True,help="factorized_beta",)
    parser.add_argument("--gauss_annealing",default="gap_stoc",type=str,help="factorized_annealing",)
    parser.add_argument("--gauss_tr",action="store_true",help="gauss_tr")

    args = parser.parse_args(argv)
    return args