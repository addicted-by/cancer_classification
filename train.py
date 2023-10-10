import pandas as pd
import torch
from cancer_classification.logger import setup_custom_logger
from cancer_classification.training.trainer import Trainer
from cancer_classification.utils import load_csv, load_thumbnails_data
from cancer_classification.utils.args import parse_arguments
from cancer_classification.utils.config import load_config
from cancer_classification.utils.datasets import ThumbnailsDataset
from torch.utils.data import DataLoader


logger = setup_custom_logger(__name__)


def main():
    default_arguments = "./configs/base/default_arguments.yaml"
    args = parse_arguments(default_arguments)
    config = args.config

    print(config)

    config = load_config(args.config, default_arguments=default_arguments)

    batch_size = config["trainer"]["batch_size"]
    logger.info(f"Batch size: {batch_size}")

    trainer = Trainer(config)
    load_thumbnails_data()
    load_csv()
    df = pd.read_csv("./data/train.csv")
    df = df[~df["is_tma"]]
    train_set = ThumbnailsDataset(df, mode="train")
    train_set, val_set = torch.utils.data.random_split(train_set, [453, 60])
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    logger.info(
        f"""
        Train loader size: {len(train_loader)},\t
        Validation loader size: {len(val_loader)}
        """
    )

    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
