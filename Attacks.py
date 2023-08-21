import math
import torch
import torch.nn.functional as F
import foolbox as fb


class FGSM:
    def __init__(self, model, epsilon=0.05, device="cpu", bounds=(-3, 3)):
        self.attack = fb.attacks.LinfFastGradientAttack()
        self.fmodel = fb.models.PyTorchModel(model, bounds, device=device)
        self.epsilon = epsilon

    def __call__(self, data, target):
        raw, perturbed, success = self.attack(self.fmodel, data, target, epsilons=self.epsilon)
        return perturbed


class IFGSM:
    def __init__(self, model, epsilon=0.05, alpha=0.02, iters=4, device="cpu", bounds=(-3, 3)):
        self.attack = FGSM(model, epsilon=alpha, device=device, bounds=bounds)
        self.epsilon = epsilon
        self.iters = iters

    def __call__(self, data, target):
        original = data.detach()
        for _ in range(self.iters):
            data = torch.clamp((self.attack(data, target) - original), -self.epsilon, self.epsilon) + original
        return data


class PGD:
    def __init__(self, model, epsilon=0.031, rel_stepsize=0.01/0.3, abs_stepsize=None, steps=40, device="cpu",
                 bounds=(-3, 3)):
        if abs_stepsize is None:
            self.attack = fb.attacks.PGD(rel_stepsize=rel_stepsize, steps=steps)
        else:
            self.attack = fb.attacks.PGD(abs_stepsize=abs_stepsize, steps=steps)
        self.fmodel = fb.models.PyTorchModel(model, bounds, device=device)
        self.epsilon = epsilon

    def __call__(self, data, target):
        raw, perturbed, success = self.attack(self.fmodel, data, target, epsilons=self.epsilon)
        return perturbed


class DeepFool:
    def __init__(self, model, epsilon=0.03, steps=50, candidates=9, overshoot=0.02, loss="logits", bounds=(-3, 3),
                 device="cpu"):
        self.attack = fb.attacks.LinfDeepFoolAttack(steps=steps, candidates=candidates,
                                                    overshoot=overshoot, loss=loss)
        self.fmodel = fb.models.PyTorchModel(model, bounds, device=device)
        self.epsilon = epsilon

    def __call__(self, data, target):
        assert data.size(0) > 1, "DeepFool does not work for batches of size 1"
        raw, perturbed, success = self.attack(self.fmodel, data, target, epsilons=self.epsilon)
        return perturbed


class AttackDataloader:
    def __init__(self, dataloader, attack, num_batches=-1, device="cpu"):
        self.batch_size = None
        self.data = []
        self.origin_images = []
        self.device = torch.device(device)
        for inpt, target in dataloader:
            if self.batch_size is None:
                self.batch_size = inpt.size(0)
            inpt = inpt.to(self.device)
            self.origin_images.append(inpt)
            target = target.to(self.device)
            self.data.append((attack(inpt, target), target))
            if 0 < num_batches <= len(self.data):
                break

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def save(self, filename):
        torch.save((self.data, self.origin_images), filename)

    def load(self, filename):
        self.data, self.origin_images = torch.load(filename)
        self.batch_size = self.data[0][0].size(0)
        for img in self.origin_images:
            img.to(self.device)
        for d in self.data:
            d[0].to(self.device)
            d[1].to(self.device)
