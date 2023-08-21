import random
import torch
import Analysis
import UniformQuantize
import GaussianQuantize
import LogQuantize
import numpy as np
from math import log10, ceil, floor
from sklearn.cluster import KMeans


def quantize_uniform_deprecated(model, layers=None, num_bits=8, quantize_activations=False):
    weights = torch.empty(0)
    if layers is None:
        weights = Analysis.get_weights(model, layers)
        if quantize_activations:
            weights = torch.cat((weights, Analysis.get_activations(model, layers)))
    quantized = UniformQuantize.quantize(weights, num_bits)
    q_levels = torch.Tensor(list(set(quantized.tolist())))
    for param in model.parameters():
        if param.data.dim() == 2:
            for i in range(len(param.data[0])):
                param.data[0][i] = q_levels[torch.argmin((q_levels - param.data[0][i]).abs())]
        elif param.data.dim() == 1 and quantize_activations:
            for i in range(len(param.data)):
                param.data[i] = q_levels[torch.argmin((q_levels - param.data[i]).abs())]


def quantize_uniform(model, num_bits=8, layers=None):
    if isinstance(num_bits, list):
        assert isinstance(layers, list), "Defining bit-width per layer requires layers to be passed"
        assert len(num_bits) == len(layers), "Given number of bit-widths and layers must match"
    signal = 0
    quantization_noise = 0
    if layers is None:
        for param in model.parameters():
            if param.data.dim() > 1:
                param_quantized = UniformQuantize.quantize(param.data, num_bits=num_bits)
                signal += torch.sum(torch.pow(param.data, 2))
                quantization_noise += torch.sum(torch.pow(param.data - param_quantized, 2))
                param.data = param_quantized
    else:
        for layer_index, layer in enumerate(layers):
            for param in layer.parameters():
                if param.data.dim() > 1:
                    if isinstance(num_bits, list):
                        param_quantized = UniformQuantize.quantize(param.data, num_bits=num_bits[layer_index])
                    else:
                        param_quantized = UniformQuantize.quantize(param.data, num_bits=num_bits)
                    signal += torch.sum(torch.pow(param.data, 2))
                    quantization_noise += torch.sum(torch.pow(param.data - param_quantized, 2))
                    param.data = param_quantized
    sqnr = signal / quantization_noise
    return log10(sqnr) * 10


def quantize_gaussian(model, num_bits=8, layers=None, quantize_activations=False, distribution="gaussian",
                      step_size_func=None, include_mean=True, include_zero=False):
    return GaussianQuantize.quantize_model(model, num_bits, layers, distribution=distribution,
                                           step_size_func=step_size_func, include_zero=include_zero)


def quantize_log(model, num_bits=8, layers=None):
    return LogQuantize.quantize_model(model, num_bits, layers)


def cross_layer_bit_width_optimization(layers, target_bit_width, quantization_efficiency=2.0, min_bits=1, max_bits=16):
    assert min_bits <= target_bit_width <= max_bits, \
        "Target bit-width has to be within bounds [" + str(min_bits) + ", " + str(max_bits) + "]"
    # check if list of layers is given or entire model
    if len(layers) > 1:
        # bit-widths are calculated in relation to each other, we chose the first layer as the base of this relation
        base = layers[0]
        layers = layers[1:]
        # calculate amount of quantized parameters in the base layer
        num_params_base = 0
        for param in base.parameters():
            if param.data.dim() > 1:
                num_params_base += torch.flatten(param).size(0)
        bit_width_diffs = [0]
        param_sizes = [num_params_base]
        for layer in layers:
            # calculate amount of quantized parameters in each layer
            num_params2 = 0
            for param in layer.parameters():
                if param.data.dim() > 1:
                    num_params2 += torch.flatten(param).size(0)
            # calculate relative bit-width for each layer
            bit_width_diffs.append((10 * log10(num_params2 / num_params_base)) / quantization_efficiency)
            param_sizes.append(num_params2)
    # if entire model is given, apply CLBWO to all parameters
    else:
        parameters = [param for param in layers[0].parameters()]
        base = parameters[0]
        parameters = parameters[1:]
        num_params_base = torch.flatten(base).size(0)
        bit_width_diffs = [0]
        param_sizes = [num_params_base]
        for param in parameters:
            num_params2 = torch.flatten(param).size(0)
            bit_width_diffs.append((10 * log10(num_params2 / num_params_base))
                                   / quantization_efficiency)
            param_sizes.append(num_params2)
    # some initial bit-width for the base layer is necessary
    base_bit_width = 32
    average_bit_width = max_bits
    while True:
        # calculate bit-widths for each layer
        absolute_bit_widths = [round(base_bit_width - diff) for diff in bit_width_diffs]
        # bit-widths can be confined to some reasonable bounds
        for i in range(len(absolute_bit_widths)):
            absolute_bit_widths[i] = max(min_bits, absolute_bit_widths[i])
            absolute_bit_widths[i] = min(max_bits, absolute_bit_widths[i])
        # average bit-width is important for comparison with target bit-width
        average_bit_width = sum([bit_width * num_params for bit_width, num_params in zip(absolute_bit_widths, param_sizes)]) \
                            / sum(param_sizes)
        # if target bit-width is reached, return
        if average_bit_width <= target_bit_width:
            return absolute_bit_widths, average_bit_width
        # otherwise, reduce base bit-width by difference between average and target
        # because average is rounded up, this will always reach target if target is larger than min_bits
        else:
            base_bit_width -= ceil(average_bit_width) - target_bit_width


def clbwo_noise(layers, target_bit_width, min_bits=1, max_bits=16):
    accs = [0.2238, 0.8613, 0.8675, 0.8684, 0.8683, 0.2722]
    assert min_bits <= target_bit_width <= max_bits, \
        "Target bit-width has to be within bounds [" + str(min_bits) + ", " + str(max_bits) + "]"
    bit_width_diffs = []
    param_sizes = []
    for i, layer in enumerate(layers):
        # calculate amount of quantized parameters in each layer
        num_params2 = 0
        for param in layer.parameters():
            if param.data.dim() > 1:
                num_params2 += torch.flatten(param).size(0)
        # calculate relative bit-width for each layer
        bit_width_diffs.append(round(torch.log2(torch.tensor(accs[i]/accs[0])).item()))
        param_sizes.append(num_params2)
    # some initial bit-width for the base layer is necessary
    base_bit_width = 32
    average_bit_width = max_bits
    while True:
        # calculate bit-widths for each layer
        absolute_bit_widths = [round(base_bit_width - diff) for diff in bit_width_diffs]
        # bit-widths can be confined to some reasonable bounds
        for i in range(len(absolute_bit_widths)):
            absolute_bit_widths[i] = max(min_bits, absolute_bit_widths[i])
            absolute_bit_widths[i] = min(max_bits, absolute_bit_widths[i])
        # average bit-width is important for comparison with target bit-width
        average_bit_width = sum(
            [bit_width * num_params for bit_width, num_params in zip(absolute_bit_widths, param_sizes)]) \
                            / sum(param_sizes)
        if average_bit_width > target_bit_width:
            base_bit_width -= ceil(average_bit_width) - target_bit_width
        else:
            param_sizes_sorted = torch.argsort(torch.tensor(param_sizes))
            changed = True
            while changed:
                changed = False
                for l_i in param_sizes_sorted:
                    layer_index = int(l_i.item())
                    # increase layers only up to the upper bound
                    if absolute_bit_widths[layer_index] < max_bits:
                        # only increase if bit-width target would not be exceeded
                        layer_size = param_sizes[layer_index] / sum(param_sizes)
                        if target_bit_width - average_bit_width >= param_sizes[layer_index] / sum(param_sizes):
                            changed = True
                            absolute_bit_widths[layer_index] += 1
                            average_bit_width += param_sizes[layer_index] / sum(param_sizes)
            # bit-width should now be as close as possible to target
            return absolute_bit_widths, average_bit_width


def clbwo_combined(layers, target_bit_width, quantization_efficiency=2.0, min_bits=1, max_bits=16):
    assert min_bits <= target_bit_width <= max_bits, \
        "Target bit-width has to be within bounds [" + str(min_bits) + ", " + str(max_bits) + "]"
    accs = [0.2238, 0.8613, 0.8675, 0.8684, 0.8683, 0.2722]
    # bit-widths are calculated in relation to each other, we chose the first layer as the base of this relation
    base = layers[0]
    layers = layers[1:]
    # calculate amount of quantized parameters in the base layer
    num_params_base = 0
    for param in base.parameters():
        if param.data.dim() > 1:
            num_params_base += torch.flatten(param).size(0)
    bit_width_diffs_struct = [0]
    bit_width_diffs_infl = [0]
    param_sizes = [num_params_base]
    for i, layer in enumerate(layers):
        # calculate amount of quantized parameters in each layer
        num_params2 = 0
        for param in layer.parameters():
            if param.data.dim() > 1:
                num_params2 += torch.flatten(param).size(0)
        # calculate relative bit-width for each layer
        bit_width_diffs_struct.append((10 * log10(num_params2 / num_params_base)) / quantization_efficiency)
        bit_width_diffs_infl.append(round(torch.log2(torch.tensor(accs[i+1]/accs[0])).item()))
        param_sizes.append(num_params2)
    # some initial bit-width for the base layer is necessary
    base_bit_width = 32
    average_bit_width = max_bits
    while True:
        # calculate bit-widths for each layer
        absolute_bit_widths = [round(base_bit_width - ((bw_struct + bw_infl)/2))
                               for bw_struct, bw_infl in zip(bit_width_diffs_struct, bit_width_diffs_infl)]
        # bit-widths can be confined to some reasonable bounds
        for i in range(len(absolute_bit_widths)):
            absolute_bit_widths[i] = max(min_bits, absolute_bit_widths[i])
            absolute_bit_widths[i] = min(max_bits, absolute_bit_widths[i])
        # average bit-width is important for comparison with target bit-width
        average_bit_width = sum(
            [bit_width * num_params for bit_width, num_params in zip(absolute_bit_widths, param_sizes)]) \
                            / sum(param_sizes)
        # reduce base bit-width until average bit-width is below target
        if average_bit_width > target_bit_width:
            # because average is rounded up, this will always reach target if target is larger than min_bits
            base_bit_width -= ceil(average_bit_width) - target_bit_width
        # increase bit-width to get as close as possible to target
        else:
            # increase most influential layers first
            param_sizes_sorted = torch.argsort(torch.tensor(param_sizes))
            # continue until no further increases can be done
            changed = True
            while changed:
                changed = False
                for l_i in param_sizes_sorted:
                    layer_index = int(l_i.item())
                    # increase layers only up to the upper bound
                    if absolute_bit_widths[layer_index] < max_bits:
                        # only increase if bit-width target would not be exceeded
                        layer_size = param_sizes[layer_index] / sum(param_sizes)
                        if target_bit_width - average_bit_width >= param_sizes[layer_index] / sum(param_sizes):
                            changed = True
                            absolute_bit_widths[layer_index] += 1
                            average_bit_width += param_sizes[layer_index] / sum(param_sizes)
            # bit-width should now be as close as possible to target
            return absolute_bit_widths, average_bit_width


def clbwo_influence(layers, target_bit_width, min_bits=1, max_bits=16):
    assert min_bits <= target_bit_width <= max_bits, \
        "Target bit-width has to be within bounds [" + str(min_bits) + ", " + str(max_bits) + "]"
    accs = [0.2238, 0.8613, 0.8675, 0.8684, 0.8683, 0.2722]
    max_acc = max(accs)
    rel_accs = [max_acc / acc for acc in accs]
    influence = [rel_acc / sum(rel_accs) for rel_acc in rel_accs]
    param_sizes = []
    for layer in layers:
        num_params = 0
        for param in layer.parameters():
            if param.data.dim() > 1:
                num_params += torch.flatten(param).size(0)
        param_sizes.append(num_params)
    absolute_bit_widths = [floor((target_bit_width * len(layers)) * infl) for infl in influence]
    average_bit_width = calc_average_bit_width(layers, absolute_bit_widths)
    # inscrease bit width until target is reached
    param_sizes_sorted = torch.argsort(torch.tensor(influence))
    changed = True
    while changed:
        changed = False
        for l_i in param_sizes_sorted:
            layer_index = int(l_i.item())
            # increase layers only up to the upper bound
            if absolute_bit_widths[layer_index] < max_bits:
                # only increase if bit-width target would not be exceeded
                if target_bit_width - average_bit_width >= param_sizes[layer_index] / sum(param_sizes):
                    changed = True
                    absolute_bit_widths[layer_index] += 1
                    average_bit_width += param_sizes[layer_index] / sum(param_sizes)
    return absolute_bit_widths, average_bit_width



def calc_average_bit_width(layers, abs_bit_widths):
    param_sizes = []
    for layer in layers:
        num_params = 0
        for param in layer.parameters():
            num_params += torch.flatten(param).size(0)
        param_sizes.append(num_params)
    return sum([bit_width * num_params for bit_width, num_params in zip(abs_bit_widths, param_sizes)]) / \
        sum(param_sizes)


def cluster_bit_widths(bit_widths, n_clusters=3):
    two_dim_bit_width = [(n, n) for n in bit_widths]
    np_bit_widths = np.array(two_dim_bit_width)
    kmeans = KMeans(n_clusters=3).fit(np_bit_widths)
    return [round(kmeans.cluster_centers_[label][0]) for label in kmeans.labels_]


def calc_share_of_params(layers):
    param_sizes = []
    for layer in layers:
        num_params = 0
        for param in layer.parameters():
            num_params += torch.flatten(param).size(0)
        param_sizes.append(num_params)
    return [param_size / sum(param_sizes) for param_size in param_sizes]


def reduce_bits_noise(layers, bit_widths, accs, reduction=0.5, min_bits=1):
    assert len(layers) == len(bit_widths) and len(bit_widths) == len(accs), ("lengths of layers, bit-widths and accs "
                                                                             "must match")
    reduced_bit_widths = [bit_width for bit_width in bit_widths]
    current_average_bit_width = calc_average_bit_width(layers, bit_widths)
    target_average_bit_width = current_average_bit_width - reduction
    relative_sizes = calc_share_of_params(layers)
    relative_sizes_args_sorted = torch.argsort(torch.tensor(accs), descending=True)
    while True:
        changed = False
        for index in relative_sizes_args_sorted:
            if (reduced_bit_widths[index] > min_bits and
                    current_average_bit_width - relative_sizes[index] > target_average_bit_width):
                reduced_bit_widths[index] -= 1
                current_average_bit_width -= relative_sizes[index]
                changed = True
        if not changed:
            break
    return reduced_bit_widths, current_average_bit_width


def reduce_bits_descending(layers, bit_widths, reduction=0.5, min_bits=1):
    reduced_bit_widths = [bit_width for bit_width in bit_widths]
    current_average_bit_width = calc_average_bit_width(layers, bit_widths)
    target_average_bit_width = current_average_bit_width - reduction
    relative_sizes = calc_share_of_params(layers)
    relative_sizes_args_sorted = torch.argsort(torch.tensor(relative_sizes), descending=True)
    while True:
        changed = False
        for index in relative_sizes_args_sorted:
            if (reduced_bit_widths[index] > min_bits and
                    current_average_bit_width - relative_sizes[index] > target_average_bit_width):
                reduced_bit_widths[index] -= 1
                current_average_bit_width -= relative_sizes[index]
                changed = True
        if not changed:
            break
    return reduced_bit_widths, current_average_bit_width

def reduce_bits_ascending(layers, bit_widths, reduction=0.5, min_bits=1):
    reduced_bit_widths = [bit_width for bit_width in bit_widths]
    current_average_bit_width = calc_average_bit_width(layers, bit_widths)
    target_average_bit_width = current_average_bit_width - reduction
    relative_sizes = calc_share_of_params(layers)
    relative_sizes_args_sorted = torch.argsort(torch.tensor(relative_sizes), descending=False)
    while True:
        changed = False
        for index in relative_sizes_args_sorted:
            if (reduced_bit_widths[index] > min_bits and
                    current_average_bit_width - relative_sizes[index] > target_average_bit_width):
                reduced_bit_widths[index] -= 1
                current_average_bit_width -= relative_sizes[index]
                changed = True
        if not changed:
            break
    return reduced_bit_widths, current_average_bit_width


def reduce_bits_random(layers, bit_widths, reduction=0.5, min_bits=1):
    reduced_bit_widths = [bit_width for bit_width in bit_widths]
    current_average_bit_width = calc_average_bit_width(layers, bit_widths)
    target_average_bit_width = current_average_bit_width - reduction
    relative_sizes = calc_share_of_params(layers)
    while True:
        available_layers_indexes = []
        for index, _ in enumerate(layers):
            if (relative_sizes[index] <= current_average_bit_width - target_average_bit_width and
                    reduced_bit_widths[index] > min_bits):
                available_layers_indexes.append(index)
        if len(available_layers_indexes) > 0:
            selection = random.randrange(0, len(available_layers_indexes))
            reduced_bit_widths[available_layers_indexes[selection]] -= 1
            current_average_bit_width -= relative_sizes[available_layers_indexes[selection]]
        else:
            return reduced_bit_widths, current_average_bit_width
