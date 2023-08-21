from torchvision import transforms


def basic():
    return {"train": transforms.ToTensor(), "test": transforms.ToTensor()}


def lsqplus():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))])
    return {"train": train_transform, "test": test_transform}


def trades():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    test_transform = transforms.Compose([
        transforms.ToTensor()])
    return {"train": train_transform, "test": test_transform}