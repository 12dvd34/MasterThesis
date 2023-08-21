import Preprocessing
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def get_train_dataloader(batch_size=32, shuffle=True, preprocessing=Preprocessing.lsqplus()):
    return DataLoader(datasets.CIFAR10(root="data", train=True, download=True, transform=preprocessing["train"]),
                      batch_size=batch_size, shuffle=shuffle)


def get_test_dataloader(batch_size=1, shuffle=False, preprocessing=Preprocessing.lsqplus()):
    return DataLoader(datasets.CIFAR10(root="data", train=False, download=True, transform=preprocessing["test"]),
                      batch_size=batch_size, shuffle=shuffle)
