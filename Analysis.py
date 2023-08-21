import torch
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.manifold import TSNE


def get_weights(model, layers=None):
    weights = torch.empty(0)
    if layers is None:
        for param in model.parameters():
            if param.data.dim() > 1:
                weights = torch.cat((weights, torch.flatten(param.data)))
    else:
        for layer in layers:
            for param in layer.parameters():
                if param.data.dim() > 1:
                    weights = torch.cat((weights, torch.flatten(param.data)))
    return weights


def get_activations(model, layers=None):
    activations = torch.empty(0)
    if layers is None:
        for param in model.parameters():
            if param.data.dim() == 1:
                activations = torch.cat((activations, torch.flatten(param.data)))
    else:
        for layer in layers:
            for param in layer.parameters():
                if param.data.dim() == 1:
                    activations = torch.cat((activations, torch.flatten(param.data)))
    return activations


def get_distribution(data):
    return norm.fit(data.detach().cpu().numpy())


def analyse_distributions(model, layers_list):
    layer_results = []
    for layers in layers_list:
        weight_distribution = get_distribution(get_weights(model, layers))
        activation_distribution = get_distribution(get_activations(model, layers))
        layer_results.append((weight_distribution[0], weight_distribution[1],
                              activation_distribution[0], activation_distribution[1]))
    return torch.tensor(layer_results)


def get_parameters_diff(params1, params2, relative=False):
    if relative:
        return torch.mean(torch.abs((params1 - params2) / params1))
    else:
        return torch.mean(torch.abs(params1 - params2))


def analyse_parameter_diffs(model1, model2, layers_list1, layers_list2, relative=False):
    results = []
    for i in range(len(layers_list1)):
        weights1 = get_weights(model1, layers_list1[i])
        weights2 = get_weights(model2, layers_list2[i])
        activations1 = get_activations(model1, layers_list1[i])
        activations2 = get_activations(model2, layers_list2[i])
        weights_diff = get_parameters_diff(weights1, weights2, relative)
        activations_diff = get_parameters_diff(activations1, activations2, relative)
        results.append((weights_diff, activations_diff))
    return torch.tensor(results)


def dim_reduce(parameters):
    return torch.tensor(TSNE().fit_transform(parameters.numpy()))


def get_feature_clusters(features, labels):
    clusters = []
    for i in range(10):
        datapoints = []
        for index, a in enumerate(features):
            if labels[index] == i:
                datapoints.append(a)
        datapoints_tensor = torch.stack(datapoints)
        centroid = torch.mean(datapoints_tensor, dim=0)
        distances = []
        for datapoint in datapoints_tensor:
            distances.append(F.pdist(torch.stack((centroid, datapoint))))
        mean_distance = torch.mean(torch.stack(distances))
        clusters.append({"centroid": centroid, "mean_distance": mean_distance})
    return clusters
