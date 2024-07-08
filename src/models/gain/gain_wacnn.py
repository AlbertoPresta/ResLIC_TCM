from ..reference.wacnn import WACNN
import torch 


def ste_round(x):
    return (torch.round(x) - x).detach() + x

class gain_WACNN(WACNN):
    """CNN based model"""

    def __init__(self, N=192, M=320,lmbda_list = [0.05], **kwargs):
        super().__init__(N = N, M = M,**kwargs)


        self.lmbda = lmbda_list #[0.0483, 0.025,0.013,0.0067,0.0035]  # mxh add from HUAWEI CVPR2021 Gained...

        # Condition on Latent y, so the gain vector length M
        # e.g.: self.levels = 6 means we have 6 pairs gain vectors corresponding to 6 level RD performance
        # treat all channels the same in initialization
        self.levels = len(self.lmbda) # 8
        self.Gain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
        self.InverseGain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
        self.HyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)
        self.InverseHyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)



    def print_information(self):
        model_tr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_fr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad== False)
        print(" trainable parameters: ",model_tr_parameters)
        print(" freeze parameters: ", model_fr_parameters)
        return model_tr_parameters

    def forward(self, x, tr = True, lv = 0):
        y = self.g_a(x)
        y_shape = y.shape[2:]
        y = y * torch.abs(self.Gain[ lv]).unsqueeze(0).unsqueeze(2).unsqueeze(3) 


        z = self.h_a(y)
        z = z * torch.abs(self.HyperGain[ lv]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset
        z_hat = z_hat * torch.abs(self.InverseHyperGain[lv]).unsqueeze(0).unsqueeze(2).unsqueeze(3)

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

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_hat = y_hat * torch.abs(self.InverseGain[ lv]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
