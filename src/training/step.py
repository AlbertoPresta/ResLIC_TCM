import torch
from utils.helper import AverageMeter, compute_msssim, compute_psnr
import wandb

def train_one_epoch(counter, model, criterion, train_dataloader, optimizer,  epoch, clip_max_norm, type='mse', annealing_strategy = None, aux_optimizer = None, wandb_log = False):
    model.train()
    device = next(model.parameters()).device


    if wandb_log:
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        mse_loss = AverageMeter()

    for i, d in enumerate(train_dataloader):
        counter += 1
        d = d.to(device)
        optimizer.zero_grad()
        if aux_optimizer is not None:
            aux_optimizer.zero_grad()

        out_net = model(d)

        if annealing_strategy is not None:
            gap = out_net["gap"]
            if annealing_strategy.type=="random":
                annealing_strategy.step(gap = gap)
                model.gaussian_conditional.stanh.beta = annealing_strategy.beta
            else: 
                lss = out_criterion["loss"].clone().detach().item()
                annealing_strategy.step(gap, epoch, lss)
                model.gaussian_conditional.stanh.beta = annealing_strategy.beta
            

            if wandb_log:
                wand_dict = {
                    "general_data/":counter,
                    "general_data/gaussian_beta: ": model.gaussian_conditional.stanh.beta
                }  
                wandb.log(wand_dict)

            

        

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()


        if wandb_log:
            loss.update(out_criterion["loss"].clone().detach())
            mse_loss.update(out_criterion["mse_loss"].clone().detach())
            bpp_loss.update(out_criterion["bpp_loss"].clone().detach())

            
            wand_dict = {
                "train_batch": counter,
                "train_batch/losses_batch": out_criterion["loss"].clone().detach().item(),
                "train_batch/bpp_batch": out_criterion["bpp_loss"].clone().detach().item(),
                "train_batch/mse":out_criterion["mse_loss"].clone().detach().item(),
            }
            wandb.log(wand_dict)

        





        if aux_optimizer is not None:
            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

        if i % 1000 == 0:
            if type == 'mse':
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'

                )
            else:
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMS_SSIM loss: {out_criterion["ms_ssim_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'

                )
    if wandb_log:
        log_dict = {
            "train":epoch,
            "train/loss": loss.avg,
            "train/bpp": bpp_loss.avg,
            "train/mse": mse_loss.avg,

            }
            
        wandb.log(log_dict)
    return counter


def test_epoch(epoch, test_dataloader, model, criterion, wandb_log = True, valid = True):
    model.eval()
    device = next(model.parameters()).device


    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)


            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            psnr.update(compute_psnr(d, out_net["x_hat"]))
            ssim.update(compute_msssim(d, out_net["x_hat"]))

    print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"

        )

    if wandb_log:
        if valid is False:
            print(
                f"Test epoch {epoch}: Average losses:"
                f"\tLoss: {loss.avg:.3f} |"
                f"\tMSE loss: {mse_loss.avg * 255 ** 2 / 3:.3f} |"
                f"\tBpp loss: {bpp_loss.avg:.2f} |"
            )
            log_dict = {
            "test":epoch,
            "test/loss": loss.avg,
            "test/bpp":bpp_loss.avg,
            "test/mse": mse_loss.avg,
            "test/psnr":psnr.avg,
            "test/ssim":ssim.avg,
            }
        else:

            print(
                f"valid epoch {epoch}: Average losses:"
                f"\tLoss: {loss.avg:.3f} |"
                f"\tMSE loss: {mse_loss.avg * 255 ** 2 / 3:.3f} |"
                f"\tBpp loss: {bpp_loss.avg:.2f} |"
            )
            log_dict = {
            "valid":epoch,
            "valid/loss": loss.avg,
            "valid/bpp":bpp_loss.avg,
            "valid/mse": mse_loss.avg,
            "valid/psnr":psnr.avg,
            "valid/ssim":ssim.avg,
            }       

        wandb.log(log_dict)

    return loss.avg



    return loss.avg