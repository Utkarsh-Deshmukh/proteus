import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from theia.backbones import BACKBONE  # type: ignore


@BACKBONE.register_module
class AlexNetBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained AlexNet model
        self.alexnet = models.alexnet(pretrained=True)
        # Remove the last two fully connected layers
        self.alexnet.classifier = torch.nn.Sequential(*list(self.alexnet.classifier.children())[:-2])
        # Set the model to evaluation mode
        self.alexnet.eval()

    def preprocess_image(self, image_path):
        # Load the image
        image = Image.open(image_path)
        # Preprocess the image
        input_tensor = self.preprocess(image)
        # Add batch dimension
        input_batch = input_tensor.unsqueeze(0)
        return input_batch

    def forward(self, image):
        # add hooks to get feature maps at various stages in the network
        feature_maps = {}

        def hook_fcn(module, input, output, name):
            feature_maps[name] = output.detach()

        hooks = []
        for name, module in self.alexnet.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                hook = module.register_forward_hook(lambda self, input, output, name=name: hook_fcn(self, input, output, name))
                hooks.append(hook)

        # Forward pass through the model to extract features
        with torch.no_grad():
            outputs = self.alexnet(image)
        # Return the extracted features
        return feature_maps
