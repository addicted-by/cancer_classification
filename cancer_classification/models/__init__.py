import os
from pathlib import Path
from typing import Dict

import torch

# import torchvision
from natsort import natsorted


def get_model_by_name(config: Dict):
    if config["trainer"]["model_name"] == "test_model":
        model = torch.nn.Identity()

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
