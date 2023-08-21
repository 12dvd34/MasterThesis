import torch
import Utils
from torchvision.transforms.functional import to_pil_image


class Image:
    def __init__(self, tensor):
        self.data = Utils.normalize(tensor.view((3, 32, 32)))

    def show(self):
        to_pil_image(self.data).show()

    def show_diff(self, image):
        to_pil_image(torch.abs(self.data - image.data)).show()

    def save(self, filename="image.png"):
        to_pil_image(self.data).save(filename, "PNG")

    def save_diff(self, image, filename="diff.png"):
        to_pil_image(torch.abs(self.data - image.data)).save(filename, "PNG")
