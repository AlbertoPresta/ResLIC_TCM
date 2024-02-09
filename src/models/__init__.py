from .reference.tcm import TCM
from .stanh.tcm_stanh import TCMSTanH 
from .Balle2018 import ScaleHyperpriorStanH
from .GainBalle2018 import GainedScaleHyperprior
from .gain.gain_wacnn import gain_WACNN
from .stanh.wacnn_stanh import WACNN_stanh


models_dict = {
    "tcm":TCM,
    "stanh":TCMSTanH,
    "scale_stanh":ScaleHyperpriorStanH,
    "scale_gain":GainedScaleHyperprior,
    "wacnn_gain": gain_WACNN,
    "wacnn_stanh":WACNN_stanh
}