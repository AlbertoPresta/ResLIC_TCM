
import torch 
from ..Balle2018 import ScaleHyperprior
from entropy_models import GaussianConditionalStanh, EntropyBottleneckStanh
from compressai.entropy_models import EntropyBottleneck 
import numpy as np 
import torch.nn.functional as F
class ScaleHyperpriorStanH(ScaleHyperprior):


    def __init__(self, N, M,
                 gaussian_configuration, 
                 factorized_configuration = None,
                fact_stanh = True,  
                **kwargs):
        super().__init__(N,M,**kwargs)


        if factorized_configuration is None:
            self.factorized_configuration = gaussian_configuration
            
        else:
            self.factorized_configuration = factorized_configuration

        if fact_stanh:
            self.entropy_bottleneck = EntropyBottleneckStanh(N,factorized_configuration = self.factorized_configuration)
        else:
            self.entropy_bottleneck = EntropyBottleneck(N)

        self.gaussian_configuration = gaussian_configuration
        self.gaussian_conditional = GaussianConditionalStanh(None,
                                                            channels = N,
                                                            gaussian_configuration =self.gaussian_configuration,
                                                    )
        

    def define_permutation(self, x):
        perm = np.arange(len(x.shape)) 
        perm[0], perm[1] = perm[1], perm[0]
        inv_perm = np.arange(len(x.shape))[np.argsort(perm)] # perm and inv perm
        return perm, inv_perm 

    def print_information(self):
        print(" g_a: ",sum(p.numel() for p in self.g_a.parameters()))
        print(" h_a: ",sum(p.numel() for p in self.h_a.parameters()))
        print(" g_s: ",sum(p.numel() for p in self.g_s.parameters()))
        print(" h_s: ",sum(p.numel() for p in self.h_s.parameters()))



        print(" TRAINABLE STANH",sum(p.numel() for p in self.gaussian_conditional.stanh.parameters() if p.requires_grad))
        print(" FROZEN STANH",sum(p.numel() for p in self.gaussian_conditional.stanh.parameters() if p.requires_grad == False))
        print("entropy_bottleneck",sum(p.numel() for p in self.entropy_bottleneck.parameters()))

        model_tr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_fr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad== False)
        print(" trainable parameters: ",model_tr_parameters)
        print(" freeze parameters: ", model_fr_parameters)
        return model_tr_parameters


    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict,strict = strict)

    def unlock_only_stanh(self, g_s_tune = False):
        for p in self.parameters():
            p.requires_grad = False 

        if g_s_tune:
            for p in self.g_s.parameters():
                p.requires_grad = True
        

        if isinstance(self.entropy_bottleneck,EntropyBottleneckStanh):

            for n,p in self.entropy_bottleneck.named_parameters():
                p.requires_grad = True 
            for p in self.entropy_bottleneck.parameters():
                p.requires_grad = True 
        
        if isinstance(self.gaussian_conditional, GaussianConditionalStanh):
            for n,p in self.gaussian_conditional.stanh.named_parameters():
                p.requires_grad = True 
            for p in self.gaussian_conditional.stanh.parameters():
                p.requires_grad = True 
                   

    def compute_gap(self, inputs, y_hat, gauss = True):
        perm, _ = self.define_permutation(inputs)
        values =  inputs.permute(*perm).contiguous() # flatten y and call it values
        values = values.reshape(1, 1, -1) # reshape values      
        y_hat_p =  y_hat.permute(*perm).contiguous() # flatten y and call it values
        y_hat_p = y_hat_p.reshape(1, 1, -1) # reshape values     
        with torch.no_grad(): 
            if gauss:   
                out = self.gaussian_conditional.stanh(values,-1) 
            else:
                out = self.entropy_bottleneck.stanh(values,-1) 
            # calculate f_tilde:  
            f_tilde = F.mse_loss(values, y_hat_p)
            # calculat f_hat
            f_hat = F.mse_loss(values, out)
            gap = torch.abs(f_tilde - f_hat)
        return gap


    def forward(self, x, tr = True, lv = 0):

        self.gaussian_conditional.stanh.update_state(x.device)


        if isinstance(self.entropy_bottleneck,EntropyBottleneckStanh):
            self.entropy_bottleneck.stanh.update_state(x.device)

        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        if isinstance(self.entropy_bottleneck,EntropyBottleneckStanh):
            z_gap = self.entropy_bottleneck.quantize(z, mode ="training")
            gap_factorized = self.compute_gap(z,  z_gap, gauss=False)
        else: 
            gap_factorized = torch.tensor(0.0).to(z.device)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means = None, training = tr)
        x_hat = self.g_s(y_hat)

        y_gap = self.gaussian_conditional.quantize(y, mode ="training") #dddd
        gap_gaussian = self.compute_gap(y,  y_gap, gauss=True)


        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "gap_gaussian":gap_gaussian,
            "gap_factorized":gap_factorized
        }
