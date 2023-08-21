import lightning.pytorch as pl
import torch
import Optimizers
import Loss
import Analysis
import torch.nn as nn
from torchvision import models


class Resnet18Lightning(pl.LightningModule):
    def __init__(self, model=None):
        super().__init__()
        if model is None:
            self.model = models.resnet18()
        else:
            self.model = model
        self.model.fc = nn.Linear(512, 10, bias=True)
        self.loss = Loss.CrossEntropyLoss(self)
        self.attack = None
        self.optimizer = Optimizers.adam(self)
        self.adv_training_alpha = 0.5

    def training_step(self, batch, batch_idx):
        inpt, target = batch
        loss = self.loss(inpt, target)
        if self.attack is not None:
            perturbed_inpt = self.attack(inpt, target)
            adv_loss = self.loss(perturbed_inpt, target)
            loss = (self.adv_training_alpha * loss) + ((1 - self.adv_training_alpha) * adv_loss)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        inpt, target = batch
        output = self.model(inpt)
        label = torch.argmax(output, dim=1)
        size = inpt.size(dim=0)
        corr = torch.eq(label, target).sum().item()
        accuracy = corr / size
        self.log("accuracy", accuracy)

    def validation_step(self, batch, batch_idx):
        inpt, target = batch
        loss = self.loss(inpt, target)
        self.log("val_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.model(batch)

    def configure_optimizers(self):
        return self.optimizer

    def save(self, filename):
        torch.save(self.model, filename)

    def load(self, filename):
        self.model = torch.load(filename)


class FeatureExtractor:
    def __init__(self, layer, max_outputs=1000):
        layer.register_forward_hook(self.hook)
        self.max_outputs = max_outputs
        self.parameters = None

    def hook(self, _, __, layer_output):
        if self.parameters is None:
            self.parameters = layer_output
        elif self.parameters.size(0) < self.max_outputs:
            self.parameters = torch.cat((self.parameters, layer_output), dim=0)

    def get_features(self):
        return self.parameters


class NoiseInjector:
    def __init__(self, layer, scale=1):
        self.handle = layer.register_forward_hook(self._hook)
        self.scale = scale

    def remove(self):
        self.handle.remove()

    def _hook(self, _, __, output):
        # scipy returns mean and standard deviation of distribution, but we need variance
        output_distribution_variance = Analysis.get_distribution(output.flatten())[1] ** 2
        # torch.randn samples from normal distribution with mean=0 and variance=1
        # at scale=1, noise will have the same variance/deviation as the layer output
        noise_variance = output_distribution_variance * self.scale
        noise = torch.randn(output.size(), dtype=torch.float32, device=output.device) * noise_variance
        return output + noise


