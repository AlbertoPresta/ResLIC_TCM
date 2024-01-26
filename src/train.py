
import wandb
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ImageFolder 
from compressai.zoo import models


from models import TCM, TCMSTanH
 
import os

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
from training.loss import RateDistortionLoss
from utils.helper import  CustomDataParallel, configure_annealings, configure_latent_space_policy, create_savepath
from utils.optimizer import configure_optimizers
from utils.parser import parse_args
from utils.plotting import plot_sos, plot_rate_distorsion
from training.step import train_one_epoch, test_epoch, compress_with_ac
from PIL import Image
from torch.utils.data import Dataset



class TestKodakDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = [os.path.join(self.data_dir,f) for f in os.listdir(self.data_dir)]

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        #transform = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor()])
        transform = transforms.Compose([transforms.ToTensor()])
        return transform(image)

    def __len__(self):
        return len(self.image_path)





def save_checkpoint(state, is_best, filename,filename_best,very_best,epoch):


    if is_best:
        torch.save(state, filename_best)
        torch.save(state, very_best)
        if epoch > 150:
            wandb.save(very_best)
    else:
        torch.save(state, filename)

def main(argv):


    wandb.init(project="cvpr2023_stanh", entity="albipresta") 
    args = parse_args(argv)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    type = args.type

    
    save_path = os.path.join(args.save_path, str(args.lmbda))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)


    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    


    psnr_res = {}
    bpp_res = {}

    if args.freeze:
        bpp_res["our"] = [0.3055]
        psnr_res["our"] = [32.529]
    else:
        bpp_res["our"] = [100]
        psnr_res["our"] = [0]

    psnr_res["base"] =   [32.529, 30.57, 29.99]
    bpp_res["base"] =  [0.3055,0.198, 0.161]  


    train_dataset = ImageFolder(args.dataset,num_images = 16016, split="train", transform=train_transforms)
    valid_dataset = ImageFolder(args.dataset, num_images = 1600, split="test", transform=test_transforms)
    test_dataset = TestKodakDataset(data_dir="/scratch/dataset/kodak")

    filelist = test_dataset.image_path

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(device)
    device = 'cuda'

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    if args.model == "stanh":
        gaussian_configuration = configure_latent_space_policy(args)
        annealing_strategy_gaussian =  configure_annealings(gaussian_configuration)
        net = TCMSTanH(gaussian_configuration=gaussian_configuration,config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=args.N, M=320)
        net = net.to(device)
    else:
        gaussian_configuration = None
        annealing_strategy_gaussian = None
        net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=args.N, M=320)
        net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args) #ffffff
    milestones = args.lr_epoch
    print("milestones: ", milestones)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=10)

    criterion = RateDistortionLoss(lmbda=args.lmbda, type=type)

    last_epoch = 0
    if args.checkpoint != "none":  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"], strict = False)
        if False: #args.continue_train:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    counter = 0
    best_loss = float("inf")

    if args.freeze:
        net.freeze()
        aux_optimizer = None
    epoch_enc = 0
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        tp = net.print_information()
        #counter, model, criterion, train_dataloader, optimizer,  epoch, clip_max_norm, type='mse', annealing_strategy = None, aux_optimizer = None, wandb_log = False
        if tp > 0:
            counter = train_one_epoch(counter, 
                                    net,
                                    criterion,
                                    train_dataloader,
                                    optimizer,
                                    epoch,
                                    args.clip_max_norm,
                                    aux_optimizer=aux_optimizer,
                                    annealing_strategy=annealing_strategy_gaussian,
                                    type = type,
                                    wandb_log = True)
            
        print("inizio validation!!!")
        val_bpp, val_psnr,valid_loss = test_epoch(epoch, valid_dataloader, net, criterion, wandb_log=True, valid = True)
        print("fine validation")
        test_bpp, test_psnr,loss = test_epoch(epoch, test_dataloader, net, criterion, wandb_log=True, valid = False)
        print("test fine")

        #lr_scheduler.step(valid_loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        print("compress init")

        test_bpp = test_bpp.clone().detach().item()

        #net.update()
        #bpp, psnr  = compress_with_ac(net, filelist, device, epoch, baseline = False, wandb_log = True)
        #print("compression results: ",bpp,"   ",psnr)

        if is_best: #and np.abs(test_bpp - bpp_res["our"][-1])>0.01:
            

            if args.freeze:
                bpp_res["our"].append(test_bpp)
                psnr_res["our"].append(test_psnr)
            else:
                bpp_res["our"] = [test_bpp]
                psnr_res["our"]= [test_psnr]
            #model, device,epoch
            plot_sos(net,device,epoch_enc)
            print("finito primo plot")
            plot_rate_distorsion(bpp_res, psnr_res,epoch_enc)
            print("finito secondo plot")
            epoch_enc +=1

        if args.checkpoint != "none":
            check = "pret"
        else:
            check = "zero"

        # creating savepath
        name_folder = check + "_" + "_" +  args.model  + "_" + str(args.N)  + "_" + str(args.symmetry) + "_" + str(args.gauss_gp)
        cartella = os.path.join(args.save_path,name_folder)


        if not os.path.exists(cartella):
            os.makedirs(cartella)
            print(f"Cartella '{cartella}' creata con successo.") 
        else:
            print(f"La cartella '{cartella}' esiste già.")



        filename, filename_best,very_best=  create_savepath(args, epoch, cartella)


        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "args":args,
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else "none",
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                filename,
                filename_best,
                very_best,
                epoch
            )


if __name__ == "__main__":  
    main(sys.argv[1:])