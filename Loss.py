import Trades
import torch.nn as nn
from torch.nn.modules.loss import _Loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.model = module.model

    def __call__(self, input, target):
        return self.criterion(self.model(input), target)


class TradesLoss(nn.Module):
    def __init__(self, module, optimizer, step_size=0.007, epsilon=0.031, perturb_steps=10, beta=6.0, distance="l_inf"):
        super().__init__()
        self.model = module.model
        self.optimizer = optimizer["optimizer"]
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.beta = beta
        self.distance = distance

    def __call__(self, input, target):
        return Trades.trades_loss(self.model, input, target, self.optimizer, self.step_size, self.epsilon,
                                  self.perturb_steps, self.beta, self.distance)


class WRLoss(nn.Module):
    def __init__(self, module, attack, robust_weight=1, base_loss=None):
        super(WRLoss, self).__init__()
        self.model = module.model
        self.attack = attack
        self.robust_weight = robust_weight
        self.base_loss = base_loss
        if self.base_loss is None:
            self.base_loss = nn.CrossEntropyLoss()

    def forward(self, input, target):
        native_loss = self.base_loss(self.model(input), target)
        perturbed_input = self.attack(input, target)
        robust_loss = self.base_loss(self.model(perturbed_input), target)
        return native_loss + (robust_loss * self.robust_weight)
