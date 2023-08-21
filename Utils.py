import torch
import numpy


def save_tensor_to_csv(tensor):
    array = numpy.asarray(tensor)
    numpy.savetxt("tensor.csv", array, delimiter=";", fmt="%f")


def get_labels(dataloader):
    labels = []
    for d in dataloader:
        labels.append(d[1][0].item())
    return torch.tensor(labels)


def normalize(tensor):
    a = tensor - tensor.min()
    return a / a.max()
