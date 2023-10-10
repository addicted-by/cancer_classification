from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from cancer_classification.logger import setup_custom_logger
from cancer_classification.models import get_model_by_name
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .asam import ASAM, SAM


logger = setup_custom_logger(__name__)


def init_weights(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):  # expand
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.trainer_config = config["trainer"]
        self.val_config = config["validation_intermediate"]
        self.opt_cfg = self.trainer_config["opt_cfg"]
        self.scheduler = None
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )  # self.config["device"] # change
        self.minimizer = None
        self._init_model()
        if not (self.trainer_config["pretrained"] and self.trainer_config["ckpt_load"]):
            self.model.apply(init_weights)

        self._init_optimizer()
        self._init_criterion()

        self.history = defaultdict(list)

    def _init_model(self):
        self.model = get_model_by_name(self.config).to(self.device)
        print(f"Initialized model {self.model._get_name()}")
        print(self.model)

    def _init_optimizer(self):
        if self.trainer_config["optimizer"] == "adam":
            self.opt = torch.optim.Adam(self.model.parameters(), **self.opt_cfg)
        elif self.trainer_config["optimizer"] == "adamw":
            self.opt = torch.optim.AdamW(self.model.parameters(), **self.opt_cfg)
        elif self.trainer_config["optimizer"] == "rmsprop":
            self.opt = torch.optim.RMSprop(self.model.parameters(), **self.opt_cfg)
        elif self.trainer_config["optimizer"] == "sgd":
            self.opt = torch.optim.SGD(self.model.parameters(), **self.opt_cfg)
        else:
            raise NotImplementedError(
                f"Optimizer {self.trainer_config['optimizer']} is not implemented!"
            )

        if self.trainer_config["lr_scheduler"]:
            if self.trainer_config["lr_scheduler"] == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.opt, **self.trainer_config["scheduler_cfg"]
                )
            elif self.trainer_config["lr_scheduler"] == "reduce_lr":
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.opt, **self.trainer_config["scheduler_cfg"]
                )
            else:
                raise NotImplementedError(
                    f"Scheduler {self.trainer_config['lr_scheduler']} is not implemented!"
                )
        if self.trainer_config["minimizer"]:
            if self.trainer_config["minimizer"] == "asam":
                self.minimizer = ASAM(
                    self.opt, self.model, **self.trainer_config["minimizer_cfg"]
                )
            elif self.trainer_config["minimzer"] == "sam":
                self.minimizer = SAM(
                    self.opt, self.model, **self.trainer_config["minimizer_cfg"]
                )
            else:
                raise ValueError(
                    f"Minimizer {self.trainer_config['minimizer']} is not available!"
                )

    def _init_criterion(self):
        if self.trainer_config["criterion"] == "crossentropy":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(
                f"Criterion {self.trainer_config['criterion']} is not implemented!"
            )

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

        desc_pattern = "Train Epoch: {} [{}/{} ({:.0f}%)]"
        pbar = tqdm(
            loader,
            unit="batch",
            total=len(loader),
            desc=desc_pattern.format(
                epoch,
                epoch,
                self.trainer_config["n_epochs"],
                epoch / self.trainer_config["n_epochs"] * 100,
            ),
        )

        losses, accs = [], []
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            logits = self.model(data)
            loss = self.criterion(logits, target)

            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

            y_pred = logits.max(1)[1].data.cpu().numpy()
            acc = np.mean(target.cpu().numpy() == y_pred)
            accs.append(acc)
            losses.append(loss.item())
            self.history["learning_rate"].append(self.opt.param_groups[0]["lr"])

            if (batch_idx + 1) % self.trainer_config["train_log_interval"] == 0:
                self.history["train_loss"].append(
                    np.mean(losses[-self.trainer_config["train_log_interval"] :])
                )

                self.history["train_accuracy"].append(
                    np.mean(accs[-self.trainer_config["train_log_interval"] :]) * 100
                )

                pbar.set_postfix(
                    loss=self.history["train_loss"][-1],
                    accuracy=self.history["train_accuracy"][-1],
                )
        if self.scheduler:
            if self.trainer_config["scheduler"] == "reduce_lr":
                self.scheduler.step(losses[-1])
            else:
                self.scheduler.step()

    def _validate(self, loader, epoch):
        self.model.train(False)

        pbar = tqdm(loader, total=len(loader), desc=f"Validation Epoch: {epoch}: ")

        losses, accs = [], []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                logits = self.model(data)
                loss = self.criterion(logits, target).item()
                y_pred = logits.max(1)[1].data.cpu().numpy()
                acc = np.mean(target.cpu().numpy() == y_pred)
                losses.append(loss)
                accs.append(acc)

                if (batch_idx + 1) % self.trainer_config["val_log_interval"] == 0:
                    self.history["val_loss"].append(
                        np.mean(losses[-self.trainer_config["val_log_interval"] :])
                    )

                    self.history["val_accuracy"].append(
                        np.mean(accs[-self.trainer_config["val_log_interval"] :]) * 100
                    )

                    pbar.set_postfix(
                        loss=self.history["val_loss"][-1],
                        accuracy=self.history["val_accuracy"][-1],
                    )

        self.history["epoch_val_loss"].append(np.mean(losses))
        self.history["epoch_val_accuracy"].append(np.mean(accs))
