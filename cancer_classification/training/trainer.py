from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from asam import ASAM, SAM
from logger import setup_custom_logger
from models import get_model_by_name
from torch.utils.tensorboard import SummaryWriter


logger = setup_custom_logger(__name__)


def init_weights(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):  # expand
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.opt_cfg = self.config["opt_cfg"]
        self.scheduler = None
        self.device = self.config["device"]
        self.minimizer = None
        self._init_model()
        if not (
            self.config["trainer"]["pretrained"] and self.config["trainer"]["ckpt_load"]
        ):
            self.model.apply(init_weights)

        self._init_optimizer()
        self._init_criterion()

        self.history = defaultdict(list)

    def _init_model(self):
        self.model = get_model_by_name(self.config).to(self.device)
        print(f"Initialized model {self.model._get_name()}")
        print(self.model)

    def _init_optimizer(self):
        if self.config["opt"] == "adam":
            self.opt = torch.optim.Adam(self.model.parameters(), **self.opt_cfg)
        elif self.config["opt"] == "adamw":
            self.opt = torch.optim.AdamW(self.model.parameters(), **self.opt_cfg)
        elif self.config["opt"] == "rmsprop":
            self.opt = torch.optim.RMSprop(self.model.parameters(), **self.opt_cfg)
        elif self.config["opt"] == "sgd":
            self.opt = torch.optim.SGD(self.model.parameters(), **self.opt_cfg)
        else:
            raise ValueError(f"Optimizer {self.config['opt']} is not implemented!")

        if self.config["scheduler"]:
            if self.config["scheduler"] == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.opt, **self.config["scheduler_cfg"]
                )
            elif self.config["scheduler"] == "reduce_lr":
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.opt, **self.config["scheduler_cfg"]
                )
            else:
                raise ValueError(
                    f"Scheduler {self.config['scheduler']} is not implemented!"
                )
        if self.config["minimizer"]:
            if self.config["minimizer"] == "asam":
                self.minimizer = ASAM(
                    self.opt, self.model, **self.config["minimizer_cfg"]
                )
            if self.config["minimzer"] == "sam":
                self.minimizer = SAM(self.opt, self.model, **self.config["minimizer_cfg"])

    def _init_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss()

    def _init_tb_logger(self):
        experiment = datetime.now().strftime("%m%d%Y.%H%M%S")
        logdir = f"./tb_logs/{self.trainer_config['model_name']}/{experiment}"
        logger.info(f"Tensorboard dir: {logdir}")
        self.tb_logger = SummaryWriter(log_dir=logdir)

    def fit(self, train_loader, val_loader):
        exp = datetime.now().strftime("%m%d%Y.%H%M%S")
        path2save = Path(f"./ckpts/{self.trainer_config['model_name']}/")
        path2save.mkdir(parents=True, exist_ok=True)
        for epoch in range(self.trainer_config["n_epochs"]):
            self._train_epoch(train_loader, epoch)
            if self.val_config["validate"] and epoch % self.val_config["interval"] == 0:
                self._validate(val_loader, epoch)

            if (epoch + 1) % self.trainer_config["save_interval"] == 0:
                ckpt_name = (
                    f"model_{self.trainer_config['model_name']}" + f"_{exp}_{epoch+1}.pth"
                )
                logger.info(f"SAVING MODEL EPOCH {epoch+1} --> {ckpt_name}")
                torch.save(self.model.state_dict(), path2save / ckpt_name)

    def _train_epoch(self, loader, epoch):
        self.model.train(True)
        pass

    def _validate(self, loader, epoch):
        self.model.train(False)
        pass
