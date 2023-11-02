# import omegaconf
import pandas as pd

# import torch
from cancer_classification.logger import setup_custom_logger
from cancer_classification.training.trainer import Trainer
from cancer_classification.utils import load_csv, load_thumbnails_data

# from cancer_classification.utils.args import parse_arguments
from cancer_classification.utils.config import load_config
from cancer_classification.utils.datasets import ThumbnailsDataset
from hydra import compose, initialize
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


logger = setup_custom_logger(__name__)


def train(config_path, default_arguments: str = "./configs/base/default_arguments.yaml"):
    # default_arguments =
    with initialize(
        version_base=None,
        config_path="configs/alex/101023/",
        job_name="cancer_classification",
    ):
        cfg = compose(
            config_name="resnet101_pretrained_ASAM_sgd_crossentropy_bs12_reduce_lr.yaml"
        )
        print(OmegaConf.to_yaml(cfg))
    # args = parse_arguments(default_arguments)
    config = cfg

    print(config)

    config = load_config(config, default_arguments=default_arguments)

    batch_size = config["trainer"]["batch_size"]
    logger.info(f"Batch size: {batch_size}")

    load_thumbnails_data()
    load_csv()
    df = pd.read_csv("./data/train.csv")
    df = df[~df["is_tma"]]

    if config["trainer"]["classification_type"] == "onevsall":
        df["label"] = df["label"].apply(lambda row: row if row == "HGSC" else "not_HDSC")
    elif config["trainer"]["classification_type"] == "minor_class":
        df["label"] = df["label"].apply(lambda row: None if row == "HGSC" else "not_HDSC")
        df.dropna(inplace=True)
    elif not config["trainer"]["classification_type"]:
        pass
    else:
        raise NotImplementedError

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"])
    train_set = ThumbnailsDataset(
        train_df, mode="train", le_ext=config["trainer"]["classification_type"]
    )
    val_set = ThumbnailsDataset(
        val_df, mode="val", le_ext=config["trainer"]["classification_type"]
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    config["trainer"]["n_classes"] = df["label"].nunique()
    trainer = Trainer(config)

    logger.info(
        f"""
        Train loader size: {len(train_loader)},\t
        Validation loader size: {len(val_loader)}
        """
    )

    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    train()
