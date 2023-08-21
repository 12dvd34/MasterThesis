import torch
import Analysis
from math import log10


def get_quantization_levels(num_bits=8, max_val=1):
    quantization_levels = []
    for i in range(2 ** (num_bits - 1)):
        quantization_levels.append(2 ** i)
        quantization_levels.append(-(2 ** i))
    scale_factor = max_val / quantization_levels[-2]
    return torch.tensor([q_level * scale_factor for q_level in quantization_levels], dtype=torch.float32)


def quantize_model(model, num_bits=8, layers=None):
    signal = 0
    quantization_noise = 0
    if layers is None:
        for param in model.parameters():
            if param.data.dim() > 1:
                max_val = torch.max(torch.abs(param)).item()
                q_levels = get_quantization_levels(num_bits, max_val)
                # dark sorcery based on:
                # https://discuss.pytorch.org/t/calculating-index-of-nearest-value-to-a-nested-tensor/143693/2
                param_quantized = torch.take(q_levels, (q_levels.repeat(*param.data.shape, 1) -
                                                        param.data.unsqueeze(-1)).abs().argmin(dim=-1))
                # SQNR
                signal += torch.sum(torch.abs(param.data))
                quantization_noise += torch.sum(torch.abs(param.data - param_quantized))
                param.data = param_quantized
    else:
        for layer_index, layer in enumerate(layers):
            for param in layer.parameters():
                if param.data.dim() > 1:
                    max_val = torch.max(torch.abs(param)).item()
                    if isinstance(num_bits, list):
                        q_levels = get_quantization_levels(num_bits[layer_index], max_val)
                    else:
                        q_levels = get_quantization_levels(num_bits, max_val)
                    param_quantized = torch.take(q_levels, (q_levels.repeat(*param.data.shape, 1) -
                                                            param.data.unsqueeze(-1)).abs().argmin(dim=-1))
                    signal += torch.sum(torch.abs(param.data))
                    quantization_noise += torch.sum(torch.abs(param.data - param_quantized))
                    param.data = param_quantized
    sqnr = signal / quantization_noise
    return log10(sqnr) * 10, quantization_noise.item()