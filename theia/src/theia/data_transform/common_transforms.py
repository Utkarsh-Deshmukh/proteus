import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from theia.data_transform import TRANSFORM  # type: ignore


@TRANSFORM.register_module
class ImageResize(torch.nn.Module):
    def __init__(self, output_dim):
        self.channels, self.height, self.width = output_dim

    def forward(self, images):
        resized_images = []
        for img in images:
            img.resize_(self.channels, self.width, self.height)
            resized_images.append(img)
        return resized_images


@TRANSFORM.register_module
class ImageCrop(torch.nn.Module):
    def __init__(self, width, height, top=0, left=0, do_center_crop=False):
        self.height, self.width, self.top, self.left = width, height, top, left

    def forward(self, images):
        cropped_images = []
        for img in images:
            img.crop((self.left, self.top, self.left + self.width, self.top + self.height))
            cropped_images.append(img)
        return cropped_images
