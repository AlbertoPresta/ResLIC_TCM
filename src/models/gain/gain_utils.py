import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)


def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)

def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')


def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",
            state_dict,
            policy,
            dtype,
        )


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

def get_scale_table(
    min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS
):  # pylint: disable=W0622
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels)) # 为什么要先ln再求e次方，是为了更高的精度吗？


class ResBlock(nn.Module):
    def __init__(self, input_channels):
        super(ResBlock, self).__init__()

        self.channels = input_channels

        self.block = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, stride=(1, 1), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, stride=(1, 1), kernel_size=3, padding=1),
        )

    def forward(self, x):
        identity_map = x
        res = self.block(x)

        return torch.add(res, identity_map)


class NonLocalAttention(nn.Module):
    def __init__(self, input_channels):
        super(NonLocalAttention, self).__init__()

        self.channels = input_channels

        self.resBlock1_trunk = ResBlock(input_channels)
        self.resBlock2_trunk = ResBlock(input_channels)
        self.resBlock3_trunk = ResBlock(input_channels)

        self.resBlock1_attention = ResBlock(input_channels)
        self.resBlock2_attention = ResBlock(input_channels)
        self.resBlock3_attention = ResBlock(input_channels)
        self.activate_attention = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, stride=(1, 1), kernel_size=1, padding=0),
            nn.Sigmoid(),
        )


    def forward(self, x):

        trunk_branch = self.resBlock1_trunk(x)
        trunk_branch = self.resBlock2_trunk(trunk_branch)
        trunk_branch = self.resBlock3_trunk(trunk_branch)

        attention_branch = self.resBlock1_attention(x)
        attention_branch = self.resBlock2_attention(attention_branch)
        attention_branch = self.resBlock3_attention(attention_branch)
        attention_branch = self.activate_attention(attention_branch)

        out = x + trunk_branch * attention_branch

        return x


def UpConv2d(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


class SFT(nn.Module):
    def __init__(self, x_nc, prior_nc=1, ks=3, nhidden=128):
        super().__init__()
        pw = ks // 2

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(prior_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw)

    def forward(self, x, qmap):
        qmap = F.adaptive_avg_pool2d(qmap, x.size()[2:])
        actv = self.mlp_shared(qmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = x * (1 + gamma) + beta

        return out