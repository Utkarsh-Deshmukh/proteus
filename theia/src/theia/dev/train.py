"""Module to facilitate training of network."""

# pylint: disable=no-name-in-module
# pylint: disable=W0511

from config import config
from theia.losses import LOSS  # type: ignore
from theia.backbones import BACKBONE
from theia.data_transform import TRANSFORM

from theia.backbones.alexnet import AlexNetBackbone
from PIL import Image

from torchvision.transforms import transforms

if __name__ == "__main__":
    loss = LOSS.compile(config.get("loss", None))
    my_backbone = BACKBONE.compile(config.get("backbone", None))

    resizer = TRANSFORM.compile(config.get("transform_config", None)[0])

    img_path = "/home/vboxuser/Desktop/workspace/proteus/theia/src/theia/data/cat1.jpg"
    img = Image.open(img_path)
    tensor_transformer = transforms.Compose([transforms.ToTensor()])
    img_tensor = tensor_transformer(img)
    img_tensor = resizer.forward([img_tensor])

    image_features = my_backbone.extract_features(img_path)
    print("done")
