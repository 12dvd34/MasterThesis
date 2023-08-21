import torch
from math import log10
from scipy.stats import norm, laplace, uniform


DISTRIBUTIONS = ["gaussian", "laplace", "uniform"]


def get_step_size_params(distribution="gaussian"):
    assert distribution in DISTRIBUTIONS, "distribution has to be any of: " + str(DISTRIBUTIONS)
    if distribution == "gaussian":
        return 2.8371, 0.5847
    elif distribution == "laplace":
        return 2.1831, 0.6844
    elif distribution == "uniform":
        return 2.0, 0.5


def get_distribution_parameters(data, distribution="gaussian"):
    assert distribution in DISTRIBUTIONS, "distribution has to be any of: " + str(DISTRIBUTIONS)
    data = data.cpu()
    if distribution == "gaussian":
        return norm.fit(data.numpy())
    elif distribution == "laplace":
        return laplace.fit(data.numpy())
    elif distribution == "uniform":
        return uniform.fit(data.numpy())


# approximated version of the quantization levels in the "Fixed Point Quantization" paper
def get_quantization_levels(distr_params, num_bits=8, include_mean=True, include_zero=False, distribution="gaussian",
                            step_size_func=None, device=None):
    if step_size_func is None:
        step_size_func = get_step_size_params
    if device is None:
        device = torch.device("cpu")
    # magic number approximating the change of step size given in the paper
    magic_num = step_size_func(distribution)[0] * (step_size_func(distribution)[1] ** num_bits)
    # the rest is scaling the step size with the distribution (paper assumes location=0, variance=1)
    step_size = magic_num * distr_params[1]
    quantization_levels = []
    quant_level = distr_params[0]
    negative_quant_levels = []
    for _ in range(int(((2 ** num_bits) / 2) - 1)):
        quant_level -= step_size
        negative_quant_levels.append(quant_level)
    quant_level = distr_params[0]
    positive_quant_levels = []
    for _ in range(int((2 ** num_bits) / 2)):
        quant_level += step_size
        positive_quant_levels.append(quant_level)
    quantization_levels.extend(reversed(negative_quant_levels))
    quantization_levels.append(distr_params[0])
    quantization_levels.extend(positive_quant_levels)
    # check if zero should be included in quantization levels
    if include_zero:
        # get quantization levels closest to zero
        closest_level = quantization_levels[0]
        for level in quantization_levels:
            if abs(level) < abs(closest_level):
                closest_level = level
        # shift quantization levels to include zero
        for i in range(len(quantization_levels)):
            quantization_levels[i] = quantization_levels[i] - closest_level
    # check if mean should not be included
    if not include_mean:
        for i in range(len(quantization_levels)):
            # shift quantization levels to the left by half a step
            quantization_levels[i] = quantization_levels[i] - (0.5 * step_size)
    return torch.tensor(quantization_levels, dtype=torch.float32, device=device)


# select closest level for each parameter
def quantize_model(model, num_bits=8, layers=None, distribution="gaussian",
                   step_size_func=None, include_mean=True, include_zero=False):
    if isinstance(num_bits, list):
        assert isinstance(layers, list), "Defining bit-width per layer requires layers to be passed"
        assert len(num_bits) == len(layers), "Given number of bit-widths and layers must match"
    signal = 0
    quantization_noise = 0
    if layers is None:
        for param in model.parameters():
            if param.data.dim() > 1:
                distr = get_distribution_parameters(torch.flatten(param.data), distribution=distribution)
                q_levels = get_quantization_levels(distr, num_bits, distribution=distribution,
                                                   step_size_func=step_size_func, device=param.device,
                                                   include_mean=include_mean, include_zero=include_zero)
                # dark sorcery based on:
                # https://discuss.pytorch.org/t/calculating-index-of-nearest-value-to-a-nested-tensor/143693/2
                param_quantized = torch.take(q_levels, (q_levels.repeat(*param.data.shape, 1) -
                                                   param.data.unsqueeze(-1)).abs().argmin(dim=-1))
                # SQNR
                signal += torch.sum(torch.pow(param.data, 2))
                quantization_noise += torch.sum(torch.pow(param.data - param_quantized, 2))
                param.data = param_quantized
    else:
        for layer_index, layer in enumerate(layers):
            for param in layer.parameters():
                if param.data.dim() > 1:
                    distr = get_distribution_parameters(torch.flatten(param.data), distribution=distribution)
                    if isinstance(num_bits, list):
                        q_levels = get_quantization_levels(distr, num_bits[layer_index], distribution=distribution,
                                                           step_size_func=step_size_func, device=param.device)
                    else:
                        q_levels = get_quantization_levels(distr, num_bits, distribution=distribution,
                                                           step_size_func=step_size_func, device=param.device)
                    param_quantized = torch.take(q_levels, (q_levels.repeat(*param.data.shape, 1) -
                                                            param.data.unsqueeze(-1)).abs().argmin(dim=-1))
                    signal += torch.sum(torch.pow(param.data, 2))
                    quantization_noise += torch.sum(torch.pow(param.data - param_quantized, 2))
                    param.data = param_quantized
    sqnr = signal / quantization_noise
    return log10(sqnr) * 10, quantization_noise.item()
