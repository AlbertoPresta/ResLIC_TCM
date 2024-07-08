from .reference.tcm import TCM
from .stanh.tcm_stanh import TCMSTanH 
from .stanh.balle18_stanh import ScaleHyperpriorStanH
from .gain.GainBalle2018 import GainedScaleHyperprior,SCGainedHyperprior
from .gain.gain_wacnn import gain_WACNN
from .stanh.wacnn_stanh import WACNN_stanh


models_dict = {
    "tcm":TCM,
    "stanh":TCMSTanH,
    "scale_stanh":ScaleHyperpriorStanH,
    "scale_gain":GainedScaleHyperprior,
    "wacnn_gain": gain_WACNN,
    "wacnn_stanh":WACNN_stanh,
    "scale_gain_sc":SCGainedHyperprior
}


from compressai.zoo import bmshj2018_hyperprior
from utils.helper import   configure_annealings, configure_latent_space_policy
import torch


def delete_keys(state_dict):
    del state_dict["entropy_bottleneck._cdf_length"]
    del state_dict["entropy_bottleneck._quantized_cdf"]
    del state_dict["entropy_bottleneck._offset"]

    del state_dict["gaussian_conditional._cdf_length"]
    del state_dict["gaussian_conditional._quantized_cdf"]
    del state_dict["gaussian_conditional._offset"]
    del state_dict["gaussian_conditional.scale_table"]

    return state_dict


def get_model(args,device):

    if args.model == "wacnn_stanh":
        gaussian_configuration = configure_latent_space_policy(args,multi = True if len(args.lambda_list) > 1 else False )
        annealing_strategy_gaussian =  configure_annealings(gaussian_configuration[0])
        factorized_configuration = None 
        annealing_strategy_factorized = None 

        net = WACNN_stanh( N = args.N, 
                          M = args.M,
                          refinement = args.refinement,
                          gaussian_configuration=gaussian_configuration, #if len(args.lambda_list) > 1 else gaussian_configuration[0], 
                          lambda_list = args.lambda_list )
        net = net.to(device)
        return net, gaussian_configuration, annealing_strategy_gaussian, factorized_configuration, annealing_strategy_factorized

    elif args.model == "stanh":

        gaussian_configuration = configure_latent_space_policy(args)
        annealing_strategy_gaussian =  configure_annealings(gaussian_configuration)

        factorized_configuration = gaussian_configuration
        annealing_strategy_factorized = configure_annealings(gaussian_configuration)


        net = TCMSTanH(lmbda = args.lambda_list,gaussian_configuration=gaussian_configuration,config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=args.N, M=320)
        net = net.to(device)
        return net, gaussian_configuration, annealing_strategy_gaussian
    elif args.model == "scale_stanh":
        if args.checkpoint != "none":
            new_args = torch.load(args.checkpoint, map_location=device)["args"]
        else:
            new_args = args
        gaussian_configuration = configure_latent_space_policy(new_args)

        annealing_strategy_gaussian =  configure_annealings(gaussian_configuration[0])
        factorized_configuration = gaussian_configuration
        annealing_strategy_factorized = configure_annealings(gaussian_configuration[0])

        net = ScaleHyperpriorStanH(N = new_args.N, M = new_args.M,gaussian_configuration=gaussian_configuration[0])
        net = net.to(device)
        

        if args.quality != 0:

            base_m = bmshj2018_hyperprior(quality=args.quality, pretrained=True).eval().to(device)
            base_m.update()
            state_dict = base_m.state_dict()

            state_dict = delete_keys(state_dict)

            #net.update(force = True) 
            net.load_state_dict(state_dict=state_dict, strict=False)

        return net, gaussian_configuration, annealing_strategy_gaussian, factorized_configuration, annealing_strategy_factorized
    elif args.model == "scale_gain" or args.model == "scale_gain_sc":
        gaussian_configuration = None,
        annealing_strategy_gaussian = None 
        factorized_configuration = None 
        annealing_strategy_factorized = None 
        net = GainedScaleHyperprior(N = args.N, M = args.M, lmbda_list = args.lambda_list) 
        net = net.to(device)
        return net, gaussian_configuration, annealing_strategy_gaussian, factorized_configuration, annealing_strategy_factorized

    elif args.model == "wacnn_gain":
        gaussian_configuration = None,
        annealing_strategy_gaussian = None 
        factorized_configuration = None 
        annealing_strategy_factorized = None 
        net = gain_WACNN(N = args.N, M = args.M, lmbda_list= args.lambda_list )
        net = net.to(device)
        return net, gaussian_configuration, annealing_strategy_gaussian, factorized_configuration, annealing_strategy_factorized
    
    else:
        gaussian_configuration = None
        annealing_strategy_gaussian = None
        net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=args.N, M=320)
        net = net.to(device)

        return net, gaussian_configuration, annealing_strategy_gaussian