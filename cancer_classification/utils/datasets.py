import os
import pickle
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms


DATA_MODES = ["train", "val", "test"]
RESCALE_SIZE = 224


class ThumbnailsDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: transforms = None,
        mode: str = "train",
        data_path: str = "./data/train_thumbnails",
        le_ext: str = "",
    ):
        self.dataframe = dataframe
        self.transform = (
            transform
            if transform
            else transforms.Compose(
                [
                    transforms.Resize((RESCALE_SIZE, RESCALE_SIZE)),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        )
        self.files = [Path(data_path) / file for file in os.listdir(data_path)]
        self.files = {int(file.stem.split("_")[0]): file for file in self.files}

        self.mode = mode
        self.label_encoder = LabelEncoder()

        if self.mode != "test":
            self.labels = self.dataframe["label"].unique()
            self.label_encoder.fit(self.labels)

            with open(f"label_encoder{le_ext}.pkl", "wb") as le_dump_file:
                pickle.dump(self.label_encoder, le_dump_file)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_id = self.dataframe.iloc[idx, 0]

        x = Image.open(self.files[image_id])
        x = self.transform(x)

        if self.mode == "test":
            return x
        else:
            label = self.dataframe.iloc[idx, 1]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
        return x, y
