"""Config specifying all the modules needed to train."""

# Backbone config
backbone_config = {"type": "resnet50"}

# Loss config
loss_config = {"type": "DiceScore", "multiclass": True}

# Head config
head_config = {"type": "classification"}


config = {"backbone": backbone_config, "loss": loss_config, "head": head_config}
