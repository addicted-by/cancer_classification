import os
import subprocess
from pathlib import Path
from typing import Callable, Union

import gdown
from cancer_classification.logger import setup_custom_logger


logger = setup_custom_logger(__name__)

TRAIN_URL = "https://drive.google.com/uc?id=13vdoUP7DhHo8ey7Nxf_km2p8eb_h2CLt"
TEST_URL = ""  # poka ne zavezli na kaggle

DATASETS_URL = {
    "train_thumbnails": TRAIN_URL,
    # "test_thumbnails": TEST_URL
}


def load_thumbnails_data():
    DATA_PATH = Path("data")
    for dataset, url in DATASETS_URL.items():
        if not ((DATA_PATH / dataset).exists() and len(os.listdir(DATA_PATH / dataset))):
            dataset_path = DATA_PATH / f"{dataset}.tar.gz"
            logger.info("Creating dir data")
            DATA_PATH.mkdir(parents=True, exist_ok=True)
            logger.info("Loading data...")
            gdown.download(url, output=str(dataset_path))
            logger.info("Untar files...")
            cmd = f"tar xzf {dataset_path} -C {DATA_PATH}"
            subprocess.run(cmd, shell=True)

    logger.info("Data had already been collected.")
    logger.info(f"Check --> {DATA_PATH}")

    return [DATA_PATH / key for key in DATASETS_URL]


def credentials_handler(func):
    def wrapper(*args, **kwargs):
        credentials = Path("~").expanduser() / ".kaggle" / "kaggle.json"

        message = f"""
            Check the credentials. Place kaggle.json in {credentials}
        """
        assert credentials.exists(), message
        logger.info("Credentials approved. Checking dependencies...")
        try:
            import kaggle
        except ImportError:
            logger.error("pip install kaggle")
        res = func(*args, **kwargs)
        return res

    return wrapper


@credentials_handler
def load_file(file: str, competition: str = "UBC-OCEAN", path2save: str = "./data"):
    if (Path(path2save) / file).exists():
        return
    subprocess.run(
        f"kaggle competitions download -c {competition} -f {file} --path {path2save}",
        shell=True,
    )


def load_image(
    idx: Union[int, str],
    competition: str = "UBC-OCEAN",
    mode: str = "train_images",
    save_path: str = "./data",
    transforms: Callable = None,
):
    thumbnails = "thumbnails" in mode
    path2save = Path(f"{save_path}/{mode}")
    path2save.mkdir(parents=True, exist_ok=True)

    zipf = path2save / Path(f"{idx}.png.zip")
    mini = "_thumbnail" if thumbnails else ""

    img = f"{mode}/{idx}{mini}.png"
    load_file(img, competition, path2save)

    subprocess.run(f"unzip {zipf} -d {path2save}", shell=True)

    logger.info("Deleting zip...")
    zipf.unlink(missing_ok=True)

    if transforms:
        img = transforms(img)


def load_csv():
    load_file("train.csv")
    load_file("test.csv")
    load_file("sample_submission.csv")


if __name__ == "__main__":
    load_image(1020)
    load_file("train.csv")
    load_file("test.csv")
    load_file("sample_submission.csv")
    load_csv()
