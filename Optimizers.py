import torch.optim as optim


def adam(module):
    return {"optimizer": optim.Adam(module.model.parameters())}


def sgd_lsqplus(module):
    optimizer = optim.SGD(module.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[31, 61, 81], gamma=0.1)
    return {"optimizer": optimizer, "lr_scheduler": scheduler}


def sgd_trades(module):
    optimizer = optim.SGD(module.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[51, 71, 91], gamma=0.1)
    return {"optimizer": optimizer, "lr_scheduler": scheduler}