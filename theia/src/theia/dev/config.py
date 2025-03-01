"""Config specifying all the modules needed to train."""

# Backbone config
backbone_config = {"type": "AlexNetBackbone"}

# Loss config
loss_config = {"type": "DiceScore", "multiclass": True}

# Head config
head_config = {"type": "classification"}

# Trasnform config
transform_config = [
    {"type": "ImageResize", "output_dim": (3, 224, 224)},
    {"type": "ImageCrop", "width": 224, "height": 224, "top": 0, "left": 0, "do_center_crop": False},
]
config = {"backbone": backbone_config, "loss": loss_config, "head": head_config, "transform_config": transform_config}
