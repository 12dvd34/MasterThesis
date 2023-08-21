import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import norm


class Plot:
    def __init__(self):
        self.fig, self.axs = plt.subplots()

    def plot_histogram(self, data, n_bins=100, density=False, comparison=None):
        if comparison is None:
            self.axs.hist(data.numpy(), n_bins, density=density)
        else:
            min_val = torch.min(comparison).item()
            max_val = torch.max(comparison).item()
            self.axs.hist(data.numpy(), n_bins, density=density, range=(min_val, max_val))
            self.axs.set(xlim=(min_val, max_val))

    def plot_distribution(self, loc=0, scale=1, increments=0.1, width=5):
        x = [(num*scale*increments)+loc for num in range(round(-width/increments), round(width/increments))]
        y = [norm.pdf(num, loc, scale) for num in x]
        self.axs.plot(x, y)

    def plot_features(self, features, labels=None):
        assert features.size(1) == 2
        if labels is None:
            # plot everything in the same color
            x = []
            y = []
            for f in features:
                x.append(f[0])
                y.append(f[1])
            self.axs.scatter(x, y)
        else:
            # plot points colored according to the sample's label
            for i in range(10):
                x = []
                y = []
                for index, a in enumerate(features):
                    if labels[index] == i:
                        x.append(a[0])
                        y.append(a[1])
                self.axs.scatter(x, y)

    def plot_feature_clusters(self, clusters):
        self.axs.axis("equal")
        for cluster in clusters:
            circle = plt.Circle(cluster["centroid"].numpy(), radius=cluster["mean_distance"].item(), fill=False)
            self.axs.add_patch(circle)

    def show(self):
        self.fig.show()

    def save(self, filename="plot.png"):
        plt.savefig(filename, dpi=300)
