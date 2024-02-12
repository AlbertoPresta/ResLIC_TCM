import torch.nn as nn 
import torch 
import numpy as np

from typing import Any, Callable, List, Optional, Tuple, Union 
import scipy.stats
from torch import Tensor
from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
from compressai.ops import LowerBound
import torch.nn.functional as F
from quantization import SymStanH, NonSymStanH
import torchac
from .coder import _EntropyCoder, default_entropy_coder, pmf_to_quantized_cdf, _forward



class HypeEntropyModelSoS(nn.Module):

    def __init__(
        self,
         removing_mean = True,
        likelihood_bound: float = 1e-9,
        entropy_coder: Optional[str] = None,
        entropy_coder_precision: int = 16,
    ):
        super().__init__()

        if entropy_coder is None:
            entropy_coder = default_entropy_coder()
        self.entropy_coder = _EntropyCoder(entropy_coder)
        self.entropy_coder_precision = int(entropy_coder_precision)
        self.removing_mean = removing_mean
        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)

        # to be filled on update()
        self.register_buffer("_offset", torch.IntTensor())
        self.register_buffer("_quantized_cdf", torch.IntTensor())
        self.register_buffer("_cdf_length", torch.IntTensor())

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["entropy_coder"] = self.entropy_coder.name
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        self.entropy_coder = _EntropyCoder(self.__dict__.pop("entropy_coder"))

    @property
    def offset(self):
        return self._offset

    @property
    def quantized_cdf(self):
        return self._quantized_cdf

    @property
    def cdf_length(self):
        return self._cdf_length

    # See: https://github.com/python/mypy/issues/8795
    forward: Callable[..., Any] = _forward

    def transform_float_to_int(self,x):
        if x not in self.stanh.unique_values:
            raise ValueError("the actual values ",x," is not present in ",self.stanh.cum_w)
        return int((self.stanh.unique_values ==x).nonzero(as_tuple=True)[0].item())
    

    def transform_int_to_float(self,x):
        return self.stanh.unique_values[x].item()



    def transform_map(self,x,map_float_to_int):
        if x in map_float_to_int.keys():
            return map_float_to_int[x]
        else:
            keys = np.asarray(list(map_float_to_int.keys()))
            keys = torch.from_numpy(keys).to(x.device)
            i = (torch.abs(keys - x)).argmin()
            key = keys[i].item()
            return map_float_to_int[key]


    def define_permutation(self, x):
        perm = np.arange(len(x.shape)) 
        perm[0], perm[1] = perm[1], perm[0]
        inv_perm = np.arange(len(x.shape))[np.argsort(perm)] # perm and inv perm
        return perm, inv_perm   


    def quantize(self, inputs, mode,  means = None, perms = None):

        if perms is None:
            perms = self.define_permutation(inputs)
        

        inputs =  inputs.permute(*perms[0]).contiguous() # flatten y and call it values
        shape = inputs.size() 
        inputs = inputs.reshape(1, 1, -1) # reshape values
        if means is not None:
            means = means.permute(*perms[0]).contiguous()
            means = means.reshape(1, 1, -1).to(inputs.device)     

        if mode == "training":

            outputs = inputs - means if (means is not None and self.removing_mean) else inputs
            outputs = self.stanh(inputs)
            outputs = outputs + means if (means is not None and self.removing_mean) else outputs


            outputs =outputs.reshape(shape)
            outputs = outputs.permute(*perms[1]).contiguous()
            return outputs
        
        outputs = inputs.clone()


        if means is not None:
            outputs -= means


        #if outputs.shape[0] == 1:
        outputs = self.stanh( outputs, -1)  
        #else:
        #outputs = self.stanh( outputs.unsqueeze(0).unsqueeze(0), -1)

        if mode == "dequantize":
            if means is not None:
                outputs += means

            outputs =outputs.reshape(shape)
            outputs = outputs.permute(*perms[1]).contiguous()
            return outputs


        outputs =outputs.reshape(shape)
        outputs = outputs.permute(*perms[1]).contiguous()


        assert mode == "symbols", mode
        shape_out = outputs.shape
        outputs = outputs.ravel()
        map_float_to_int = self.stanh.map_sos_cdf 
        
        for i in range(outputs.shape[0]):
            outputs[i] =  self.transform_map(outputs[i], map_float_to_int)
            if i%1000==0:
                print(i)
        

        outputs = outputs.reshape(shape_out) 
        outputs = outputs.to(dtype=torch.int)   
        return outputs



    def map_to_level(self, inputs, maps, dequantize = False):
        shape_out = inputs.shape
        outputs = inputs.ravel()
        for i in range(outputs.shape[0]):
            if dequantize is False:
                outputs[i] =  self.transform_map(outputs[i], maps)
            else: 
                outputs[i] =   torch.from_numpy(np.asarray(self.transform_map(outputs[i], maps))).to(outputs.device)
        outputs = outputs.reshape(shape_out)
        outputs = outputs.int()   
        return outputs     

     
    def dequantize(self, inputs, means = None, dtype = torch.float):
        """
        we have to 
        1 -map again the integer values to the real values for each channel
        2 - ad the means  
        """
        inputs = inputs.to(dtype)
        map_int_to_float = self.stanh.map_cdf_sos
        shape_inp = inputs.shape
        inputs = inputs.ravel()
        for i in range(inputs.shape[0]):
            c = torch.tensor(map_int_to_float[inputs[i].item()],dtype=torch.float32)
            inputs[i] = c.item()
        inputs = inputs.reshape(shape_inp)

        if means is not None:
            inputs = inputs.type_as(means)
            inputs += means
        outputs = inputs.type(dtype)
        return outputs



    def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
        cdf = torch.zeros(
            (len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device
        )
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = pmf_to_quantized_cdf(prob, self.entropy_coder_precision)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf

    def _check_cdf_size(self):
        if self._quantized_cdf.numel() == 0:
            raise ValueError("Uninitialized CDFs. Run update() first")

        if len(self._quantized_cdf.size()) != 2:
            raise ValueError(f"Invalid CDF size {self._quantized_cdf.size()}")

    def _check_offsets_size(self):
        if self._offset.numel() == 0:
            raise ValueError("Uninitialized offsets. Run update() first")

        if len(self._offset.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._offset.size()}")

    def _check_cdf_length(self):
        if self._cdf_length.numel() == 0:
            raise ValueError("Uninitialized CDF lengths. Run update() first")

        if len(self._cdf_length.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._cdf_length.size()}")

    




    def retrieve_cdf(self,shapes,indexes):
        output_cdf = torch.zeros(shapes)
        output_cdf = output_cdf[:,None] + torch.zeros(self.cdf.shape[1])
        output_cdf = output_cdf.to("cpu")
        for i in range(shapes[0]):
            output_cdf[i,:] = self.cdf[indexes[i].item(),:]  
        return output_cdf 
    
    
    """
    def compress_old(self, inputs, indexes):


        symbols = inputs #[1,128,32,48]
        shape_symbols = symbols.shape


        symbols = symbols.ravel().to(torch.int16)
        indexes = indexes.ravel().to(torch.int16)

        
        symbols = symbols.to("cpu")  
        output_cdf = self.retrieve_cdf(symbols.shape, indexes)

        byte_stream = torchac.encode_float_cdf(output_cdf, symbols, check_input_bounds=True)
        c = torchac.decode_float_cdf(output_cdf, byte_stream)
        
        #if torchac.decode_float_cdf(output_cdf, byte_stream).equal(symbols) is False:
        #    raise ValueError("L'output Gaussiano codificato è diverso, qualcosa non va!")  #ssssss
        #else:
        #    print("l'immagine è ok!")
        return byte_stream, c, shape_symbols 
    """
    

    def compress(self, symbols, indexes):
        """
        Compress input tensors to char strings.

        Args:
            inputs (torch.Tensor): input tensors
            indexes (torch.IntTensor): tensors CDF indexes
            means (torch.Tensor, optional): optional tensor means
        """
        #symbols = self.quantize(inputs, "symbols", means)

        if len(symbols.size()) < 2:
            raise ValueError(
                "Invalid `inputs` size. Expected a tensor with at least 2 dimensions."
            )

        if symbols.size() != indexes.size():
            raise ValueError("`inputs` and `indexes` should have the same size.")



        strings = []

        for i in range(symbols.size(0)):
            rv = self.entropy_coder.encode_with_indexes(
                symbols[i].reshape(-1).int().tolist(),
                indexes[i].reshape(-1).int().tolist(),
                self._quantized_cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            strings.append(rv)
        return strings






   




class GaussianConditionalStanh(HypeEntropyModelSoS):

    def __init__(
        self,
        
        scale_table: Optional[Union[List, Tuple]],
        *args: Any,
        scale_bound: float = 0.11,
        tail_mass: float = 1e-9, 
        gaussian_configuration = None,    
        channels: int = 128, 
        **kwargs: Any,
    ):
        super().__init__(removing_mean = gaussian_configuration["removing_mean"], *args, **kwargs)

        if not isinstance(scale_table, (type(None), list, tuple)):
            raise ValueError(f'Invalid type for scale_table "{type(scale_table)}"')

        if isinstance(scale_table, (list, tuple)) and len(scale_table) < 1:
            raise ValueError(f'Invalid scale_table length "{len(scale_table)}"')

        if scale_table and (
            scale_table != sorted(scale_table) or any(s <= 0 for s in scale_table)
        ):
            raise ValueError(f'Invalid scale_table "({scale_table})"')

        self.tail_mass = float(tail_mass)
        if scale_bound is None and scale_table:
            scale_bound = self.scale_table[0]
        if scale_bound <= 0:
            raise ValueError("Invalid parameters")
        self.lower_bound_scale = LowerBound(scale_bound)

        self.register_buffer(
            "scale_table",
            self._prepare_scale_table(scale_table) if scale_table else torch.Tensor(),
        )

        self.register_buffer(
            "scale_bound",
            torch.Tensor([float(scale_bound)]) if scale_bound is not None else None,
        )


        self.channels = int(channels)
        self.M = int(channels)
        self.tail_mass = float(tail_mass)
        self.num_sigmoids = int(gaussian_configuration["num_sigmoids"])

        self.extrema = gaussian_configuration["extrema"]
        self.symmetry = gaussian_configuration["symmetry"]


        if self.symmetry is False: 
            self.stanh = NonSymStanH(gaussian_configuration["beta"],self.num_sigmoids, extrema = self.extrema, trainable= gaussian_configuration["trainable"])
        else:
            self.stanh = SymStanH(gaussian_configuration["beta"],self.num_sigmoids, extrema = self.extrema, trainable= gaussian_configuration["trainable"])

          
    @staticmethod
    def _prepare_scale_table(scale_table):
        return torch.Tensor(tuple(float(s) for s in scale_table))

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)


    def update_scale_table(self, scale_table):
        # Check if we need to update the gaussian conditional parameters, the
        # offsets are only computed and stored when the conditonal model is
        # updated.
        device = self.scale_table.device
        self.scale_table = self._prepare_scale_table(scale_table).to(device)
        self.update()
        return True

    
    
    def update(self, device = torch.device("cuda")):


        self.stanh.update_state(device)
        max_length = self.stanh.cum_w.shape[0]
            


        pmf_length = torch.zeros(self.scale_table.shape[0]).int().to(device) + max_length
        pmf_length = pmf_length.unsqueeze(1)

        self.stanh.define_channels_map()


        average_points = self.stanh.average_points # punti-medi per ogni livello di quantizzazione 
        distance_points = self.stanh.distance_points

        samples = self.stanh.cum_w
        samples = samples.repeat(self.scale_table.shape[0],1)
        samples = samples.to(device)

        print("dio cristo: ",samples)

        self._offset = -self.stanh.cum_w[0]


        low,up = self.define_v0_and_v1(samples, average_points, distance_points)
        low = low.to(samples.device)
        up = up.to(samples.device)


        samples_scale = self.scale_table.unsqueeze(1)  #[64,1]
        samples = samples.float()
        #samples = torch.abs(samples) # da correggerre 
        samples_scale = samples_scale.float()
        

            # adapt to non simmetric quantization steps 
        upper_pos = self._standardized_cumulative((low - samples) / samples_scale)*(samples >= 0)
        upper_neg = self._standardized_cumulative((samples + up) / samples_scale)*(samples < 0)
        lower_pos = self._standardized_cumulative((-up  - samples) / samples_scale)*(samples >= 0)
        lower_neg = self._standardized_cumulative(( samples - low) / samples_scale)*(samples < 0)
            
        upper = upper_pos + upper_neg
        lower = lower_pos + lower_neg
            
        pmf = upper - lower

        self.pmf = pmf
        self.cdf =  self.pmf_to_cdf()

        # loro 
        tail_mass = 2 * lower[:, :1]
        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        
        self._cdf_length = pmf_length + 2


    """
    def update(self):
        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        pmf_center = torch.ceil(self.scale_table * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.abs(torch.arange(max_length, device=device).int() - pmf_center[:, None])

        
        samples_scale = self.scale_table.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = self._standardized_cumulative((0.5 - samples) / samples_scale)
        lower = self._standardized_cumulative((-0.5 - samples) / samples_scale)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2
    """


    def pmf_to_cdf(self):
        cdf = self.pmf.cumsum(dim=-1)
        spatial_dimensions = self.pmf.shape[:-1] + (1,)
        zeros = torch.zeros(spatial_dimensions, dtype=self.pmf.dtype, device=self.pmf.device)
        cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
        cdf_with_0 = cdf_with_0.clamp(max=1.)
        return cdf_with_0
         

    
    def define_v0_and_v1(self, inputs, average_points, distance_points): 


        inputs_shape = inputs.shape
        inputs = inputs.reshape(-1) #.to(inputs.device) # perform reshaping 
        inputs = inputs.unsqueeze(1)#.to(inputs.device) # add a dimension
       
        average_points = average_points.to(inputs.device)
        distance_points = distance_points.to(inputs.device)
       
        
        average_points_left = torch.zeros(average_points.shape[0] + 1 ).to(inputs.device) - 1000 # 1000 è messo a caso al momento 
        average_points_left[1:] = average_points
        average_points_left = average_points_left.unsqueeze(0).to(inputs.device)
        

        average_points_right = torch.zeros(average_points.shape[0] + 1 ).to(inputs.device) + 1000 # 1000 è messo a caso al momento 
        average_points_right[:-1] = average_points
        average_points_right = average_points_right.unsqueeze(0).to(inputs.device)       
               
               
        distance_points_left = torch.cat((torch.tensor([0]).to(inputs.device),distance_points),dim = -1).to(inputs.device)
        distance_points_left = distance_points_left.unsqueeze(0).to(inputs.device)
        
        distance_points_right = torch.cat((distance_points, torch.tensor([0]).to(inputs.device)),dim = -1).to(inputs.device)
        distance_points_right = distance_points_right.unsqueeze(0).to(inputs.device)
        
        li_matrix = inputs > average_points_left # 1 if x in inputs is greater that average point, 0 otherwise. shape [__,15]
        ri_matrix = inputs <= average_points_right # 1 if x in inputs is smaller or equal that average point, 0 otherwise. shape [__,15]
        
        li_matrix = li_matrix.to(inputs.device)
        ri_matrix = ri_matrix.to(inputs.device)

        one_hot_inputs = torch.logical_and(li_matrix, ri_matrix).to(inputs.device) # tensr that represents onehot encoding of inouts tensor (1 if in the interval, 0 otherwise)
              
        one_hot_inputs_left = torch.sum(distance_points_left*one_hot_inputs, dim = 1).unsqueeze(1).to(inputs.device) #[1200,1]
        one_hot_inputs_right = torch.sum(distance_points_right*one_hot_inputs, dim = 1).unsqueeze(1).to(inputs.device) #[1200,1]
        
        
        v0 = one_hot_inputs_left.reshape(inputs_shape)#.to(inputs.device) #  in ogni punto c'è la distanza con il livello a sinistra       
        v1 = one_hot_inputs_right.reshape(inputs_shape)#.to(inputs.device) # in ogni punto c'è la distanza con il livello di destra

        return v0 , v1


    #add something
    def _likelihood(self, inputs: Tensor, scales: Tensor, means: Optional[Tensor] = None):


        average_points = self.stanh.average_points.to(inputs.device)
        distance_points = self.stanh.distance_points.to(inputs.device)

        if means is not None:
            values = inputs - means
        else:
            values = inputs
        
        #values = torch.abs(values)
        low,up = self.define_v0_and_v1(values, average_points, distance_points)
        low = low.to(inputs.device)
        up = up.to(inputs.device)


        values = values.to(inputs.device)
        #values = torch.abs(values).to(inputs.device)


        scales = self.lower_bound_scale(scales)

        upper_pos = self._standardized_cumulative((low - values) / scales)*(values >= 0)
        upper_neg = self._standardized_cumulative((values + up) / scales)*(values < 0)
        lower_pos = self._standardized_cumulative((-up  - values) / scales)*(values >= 0)
        lower_neg = self._standardized_cumulative(( values - low) / scales)*(values < 0)
        

        upper = upper_pos  + upper_neg
        lower = lower_pos + lower_neg
        
        #lower = lower_pos + lower_neg
        likelihood = upper - lower

        #upper =self._standardized_cumulative((low - values) / scales)
        #lower = self._standardized_cumulative((-up - values) / scales)

        #likelihood = upper - lower
        return likelihood







    def forward(self, values ,scales ,  training = True, means = None):


        if training is None:
            training = self.training 


        perm,inv_perm = self.define_permutation(values)

        y_hat = self.quantize(values, "training" if training else "dequantize",perms = [perm,inv_perm], means = means)
        
        likelihood = self._likelihood(y_hat, scales, means = means)#.to(x.device)  nuovo !!
        #likelihood = self._likelihood(values, scales, means = means).to(x.device)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)  
        return y_hat, likelihood 


    def build_indexes(self, scales: Tensor):
        """
        Questa funzione associa ad ogni elemento output scala l'indice corrispondende alla deviazione standard 
        one-to-one mapping tra scala e indexe
        Non è ottimale, perché si associa la scala subito più grande da una lista fissata
        1- la lista fissata mi sembra troppo estesa (serve?)
        """
        scales = self.lower_bound_scale(scales)
        indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            indexes -= (scales <= s).int()
        return indexes



    def permutation_function(self,x):
        perm = np.arange(len(x.shape))
        perm[0], perm[1] = perm[1], perm[0]
        inv_perm = np.arange(len(x.shape))[np.argsort(perm)]
        return perm, inv_perm







    def compress(self, x, indexes, perms = None, means = None ):

        if perms is None:
            perms = self.define_permutation(x)

        x = self.quantize(x, "symbols", means = means, perms = perms)
        return super().compress(x,indexes)
        #x = self.quantize(x, "symbols", means = means, perms = perms)  
        #byte_stream, c, shape_symbols  = super().compress(x, indexes)  #ddddd
        #return byte_stream, c, shape_symbols



    def decompress(self, strings, size):
        output_size = (len(strings), self._quantized_cdf.size(0), *size)
        indexes = self._build_indexes(output_size).to(self._quantized_cdf.device)
        medians = self._extend_ndims(self._get_medians().detach(), len(size))
        medians = medians.expand(len(strings), *([-1] * (len(size) + 1)))
        return super().decompress(strings, indexes, medians,0)



    """
    
    def decompress(self, byte_stream,  std,indexes, means = None):
        #outputs = super().decompress(byte_stream, output_cdf) 
        shapes = std.shape  
        cdf_shapes = std.ravel().shape
        indexes = indexes.ravel().to(torch.int16)

        output_cdf = self.retrieve_cdf(cdf_shapes,indexes)
        outputs =   torchac.decode_float_cdf(output_cdf, byte_stream)
        #print("lo shape è ---> ",outputs.shape,"     ",means.shape)
        #outputs = outputs.to("cuda")
        #means = means.to("cuda")
       
        outputs = outputs.reshape(shapes)
        means = means.reshape(shapes)

        outputs = self.dequantize(outputs, means = means)
        outputs = outputs.to("cuda")
        return outputs
    """

    def decompress(self, strings, indexes, means=None, flag=1):
        """
        Decompress char strings to tensors.

        Args:
            strings (str): compressed tensors
            indexes (torch.IntTensor): tensors CDF indexes
            means (torch.Tensor, optional): optional tensor means
        """

        if not isinstance(strings, (tuple, list)):
            raise ValueError("Invalid `strings` parameter type.")

        if not len(strings) == indexes.size(0):
            raise ValueError("Invalid strings or indexes parameters")

        if len(indexes.size()) < 2:
            raise ValueError(
                "Invalid `indexes` size. Expected a tensor with at least 2 dimensions."
            )



        if means is not None:
            if means.size()[:2] != indexes.size()[:2]:
                raise ValueError("Invalid means or indexes parameters")
            if means.size() != indexes.size():
                for i in range(2, len(indexes.size())):
                    if means.size(i) != 1:
                        raise ValueError("Invalid means parameters")

        cdf = self._quantized_cdf
        outputs = cdf.new_empty(indexes.size())

        for i, s in enumerate(strings):
            values = self.entropy_coder.decode_with_indexes(
                s,
                indexes[i].reshape(-1).int().tolist(),
                cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            outputs[i] = torch.tensor(
                values, device=outputs.device, dtype=outputs.dtype
            ).reshape(outputs[i].size())


        outputs = self.dequantize(outputs, means = means)
        return outputs


    """
    def decompress_old(self, byte_stream,  output_cdf):
        #outputs = super().decompress(byte_stream, output_cdf) 


        outputs =   torchac.decode_float_cdf(output_cdf, byte_stream)
        #outputs = outputs.to("cuda")
        return outputs
    """