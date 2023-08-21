import json
import torch
import torch.nn.functional as F
import foolbox as fb


def attack_custom(data, target, model, epsilon=0.05):
    model.zero_grad()
    data.requires_grad = True
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    return torch.clamp(data + (epsilon * data.grad.data), 0, 1)


def attack_foolbox(data, target, model, epsilon=0.05):
    attack = fb.attacks.LinfFastGradientAttack()
    fmodel = fb.models.PyTorchModel(model, (0, 1))
    raw, perturbed, success = attack(fmodel, data, target, epsilons=epsilon)
    return perturbed


class FGSMDataloader:
    def __init__(self, dataloader, num_samples, model, epsilon=0.05, attack_method=attack_custom, device="cpu"):
        self.batch_size = 1
        self.data = []
        self.device = torch.device(device)
        for inpt, target in dataloader:
            inpt = inpt.to(self.device)
            target = target.to(self.device)
            self.data.append((attack_method(inpt, target, model, epsilon), target))
            if len(self.data) >= num_samples:
                break

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
