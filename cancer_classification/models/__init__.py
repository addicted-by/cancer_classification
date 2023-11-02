import os
from pathlib import Path
from typing import Dict

import torch
import torchvision

# import torchvision
from natsort import natsorted

from .resnet18 import ResNet18


def get_model_by_name(config: Dict):
    n_classes = config["trainer"]["n_classes"]
    pretrained = config["trainer"]["pretrained"]
    if config["trainer"]["model_name"] == "resnet18":
        model = torchvision.models.resnet18(pretrained=pretrained)
        model.fc = torch.nn.Linear(512, n_classes)
        model.add_module("SoftMax", torch.nn.Softmax(dim=-1))
    elif config["trainer"]["model_name"] == "resnet101":
        model = torchvision.models.resnet101(pretrained=pretrained)

        if config["trainer"]["freeze_layers"]:
            for num, child in enumerate(model.children()):
                if num < 6:
                    for param in child.parameters():
                        param.requires_grad = False

        model.fc = torch.nn.Linear(2048, n_classes)
        model.add_module("SoftMax", torch.nn.Softmax(dim=-1))
        # add activation (NLLoss -- LogSoftMax, GumbelSoftMax)
    else:
        raise NotImplementedError(
            f"Cannot process this model name: {config['trainer']['model_name']}"
        )

    if config["trainer"]["ckpt_load"]:
        ckpt = config["trainer"]["ckpt_load"]
        if ckpt == "last":
            ckpts_dir = (
                Path(config["trainer"]["ckpt_dir"]) / config["trainer"]["model_name"]
            )
            last_ckpt = natsorted(os.listdir(ckpts_dir))[-1]
            ckpt = ckpts_dir / last_ckpt

        print(f"Loading checkpoint {ckpt}")
        err_msg = f"Checkpoint {ckpt} does not exists. Please check!"
        assert os.path.exists(ckpt), err_msg
        model.load_state_dict(torch.load(ckpt))

    return model
