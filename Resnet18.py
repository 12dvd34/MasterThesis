import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


class Resnet18():
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = models.resnet18()
        self.model.fc = nn.Linear(512, 10, bias=True)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.model.to(self.device)
        self.criterion.to(self.device)

    def train(self, dataloader, n_samples=100, epochs=10):
        for epoch in range(epochs):
            epoch_loss = 0
            for index in range(n_samples):
                self.optimizer.zero_grad()
                input, target = next(iter(dataloader))
                input = input.to(self.device)
                target = target.to(self.device)
                output = self.model(input)
                loss = self.criterion(output, target)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            print("epoch " + str(epoch) + " loss: " + str(epoch_loss / n_samples))

    def test(self, dataloader, n_samples=100):
        with torch.no_grad():
            corr = 0
            num = 0
            for input, target in dataloader:
                if num >= n_samples:
                    break
                input = input.to(self.device)
                target = target.to(self.device)
                output = self.model(input)
                label = torch.argmax(output, dim=1)
                corr += torch.eq(label, target).sum().item()
                num += dataloader.batch_size
        return corr / num

    def save(self, filename):
        torch.save(self.model, filename)

    def load(self, filename):
        self.model = torch.load(filename).to(self.device)