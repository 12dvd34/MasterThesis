from Resnet18Lightning import Resnet18Lightning
from torch.nn.utils import clip_grad_norm
import torch


class Resnet18Quantized(Resnet18Lightning):
    def __init__(self, num_bits=8, layers=None, delay=0, model=None, scale_weights=False):
        super().__init__(model=model)
        self.num_bits = num_bits
        self.delay = delay
        self.scale_weights = scale_weights
        self.layers = layers
        self.epoch = 0
        self.full_precision_params = []
        self.is_quantized = False
        if isinstance(num_bits, list):
            if isinstance(layers, list):
                assert len(num_bits) == len(layers), "Given number of bit-widths and layers must match"
            elif layers is None:
                assert len([_ for _ in self.model.parameters()]) == len(num_bits), "Bit widths are defined per layer " \
                                                                                   "but no layers are passed and the " \
                                                                                   "number of bit-widths doesn't " \
                                                                                   "equal the number of parameters " \
                                                                                   "in the model."
            else:
                raise TypeError("Expected layers to be either a list or None")
        elif isinstance(num_bits, dict):
            assert 0 in num_bits.keys(), "Expected num_bits to at least contain a '0' key"

    def on_train_epoch_end(self):
        self.epoch += 1

    def on_after_backward(self):
        # dequantize for optimizer step
        self.dequantize()

    def on_train_batch_start(self, batch, batch_idx):
        # quantize for forward pass and backprop
        self.quantize()

    def on_validation_start(self):
        if not self.is_quantized:
            self.quantize()

    def on_validation_end(self):
        self.dequantize()

    def on_test_start(self):
        if not self.is_quantized:
            self.quantize()

    def on_test_end(self):
        self.dequantize()

    def on_predict_start(self):
        if not self.is_quantized:
            self.quantize()

    def on_predict_end(self):
        self.dequantize()

    def quantize(self):
        tmp_num_bits = None
        # skip quantization if delay is set and not yet reached
        if self.epoch < self.delay:
            return
        # if dict is passed, select appropriate element
        if isinstance(self.num_bits, dict):
            # save num_bits dict to restore it after quantization
            tmp_num_bits = self.num_bits
            # keys are not guaranteed to be sorted, find the largest key that is smaller than current epoch
            max_key = 0
            for key in self.num_bits.keys():
                if self.epoch >= key > max_key:
                    max_key = key
            self.num_bits = self.num_bits[max_key]
        # save full precision weights and then quantize them
        if self.layers is None:
            # check if bit-widths have been given for every param...
            if isinstance(self.num_bits, list):
                for param_index, param in enumerate(self.model.parameters()):
                    self.full_precision_params.append(param.data.clone().detach())
                    quantized = quantize_param(param, num_bits=self.num_bits[param_index]).clone().detach()
                    if self.scale_weights:
                        param.data = quantized * (torch.mean(torch.abs(param.data.clone().detach()))
                                                  / (torch.mean(torch.abs(quantized) + 1e-6)))
                    else:
                        param.data = quantized
            # ...or the entire model
            else:
                for param in self.model.parameters():
                    self.full_precision_params.append(param.data.clone().detach())
                    quantized = quantize_param(param, num_bits=self.num_bits).clone().detach()
                    if self.scale_weights:
                        param.data = quantized * (torch.mean(torch.abs(param.data.clone().detach()))
                                                  / (torch.mean(torch.abs(quantized) + 1e-6)))
                    else:
                        param.data = quantized
        else:
            for layer_index, layer in enumerate(self.layers):
                for param in layer.parameters():
                    self.full_precision_params.append(param.data.clone().detach())
                    quantized = quantize_param(param, num_bits=self.num_bits[layer_index]).clone().detach()
                    if self.scale_weights:
                        param.data = quantized * (torch.mean(torch.abs(param.data.clone().detach()))
                                                  / (torch.mean(torch.abs(quantized) + 1e-6)))
                    else:
                        param.data = quantized
        self.is_quantized = True
        # restore num_bits dict
        if tmp_num_bits is not None:
            self.num_bits = tmp_num_bits

    def dequantize(self):
        # load full precision weights
        if self.layers is None:
            for layer_index, param in enumerate(self.model.parameters()):
                param.data = self.full_precision_params[layer_index].clone().detach()
        else:
            param_index = 0
            for layer_index, layer in enumerate(self.layers):
                for param in layer.parameters():
                    param.data = self.full_precision_params[param_index].clone().detach()
                    param_index += 1
        self.is_quantized = False
        # clear full precision params
        self.full_precision_params = []


def quantize_param(param, num_bits=8, full_scale_range=1, in_place=False):
    # reduce bit-width by 1 to account for sign bit
    num_bits = num_bits - 1
    if param.data.dim() < 2:
        return param
    scale_factor = torch.max(torch.abs(param))
    scaled = param / scale_factor
    log = torch.log2(torch.abs(scaled))
    rounded = torch.round(log).to(torch.int)
    # implement weird clipping function without loops and if statements
    # upper end
    clipped_upper_tmp = torch.clip(rounded, max=full_scale_range)
    diff_upper = torch.abs(rounded - clipped_upper_tmp)
    binary_diff_upper = torch.round(torch.clip(diff_upper, 0, 1))
    max_clip_val = full_scale_range - 1
    clipped_upper = (binary_diff_upper * max_clip_val) + ((1 - binary_diff_upper) * rounded)
    # lower end
    clipped_lower_tmp = torch.clip(clipped_upper, min=full_scale_range - pow(2, num_bits))
    diff_lower = torch.abs(clipped_upper - clipped_lower_tmp)
    binary_diff_lower = torch.round(torch.clip(diff_lower, 0, 1))
    min_clip_val = 0
    clipped = (binary_diff_lower * min_clip_val) + ((1 - binary_diff_lower) * rounded)
    # implement LogQuant function without if statement
    # this will treat numbers near 0 as 0
    # make this number bigger if too many weights get rounded down to 0
    upscale = 1000000
    not_zero = torch.round(torch.clip(torch.abs(clipped * upscale), 0, 1)).to(torch.int)
    two = torch.tensor(2, device=param.device).repeat(param.size())
    log_quant = (not_zero * torch.pow(two.to(torch.float), clipped.to(torch.float)))
    signed = log_quant * torch.sign(param)
    sqnr = torch.sum(torch.pow(scaled, 2)) / torch.sum(torch.pow(scaled - signed, 2))
    if in_place:
        param.data = signed
    else:
        return signed
