import subprocess
from pathlib import Path
from typing import Union


def credentials_handler(func):
    def wrapper(*args, **kwargs):
        credentials = Path("~").expanduser() / ".kaggle" / "kaggle.json"

        message = f"""
            Check the credentials. Place kaggle.json in {credentials}
        """
        assert credentials.exists(), message
        print("Credentials approved. Checking dependencies...")
        try:
            import kaggle
        except ImportError:
            print("pip install kaggle")
        res = func(*args, **kwargs)
        return res

    return wrapper


@credentials_handler
def load_file(file: str, competition: str = "UBC-OCEAN", path2save: str = "./data"):
    subprocess.run(
        f"kaggle competitions download -c {competition} -f {file} --path {path2save}",
        shell=True,
    )


def load_image(
    idx: Union[int, str],
    competition: str = "UBC-OCEAN",
    mode: str = "train_images",
    save_path: str = "./data",
):
    path2save = Path(f"{save_path}/{mode}")
    path2save.mkdir(parents=True, exist_ok=True)

    zipf = path2save / Path(f"{idx}.png.zip")
    load_file(f"{mode}/{idx}.png", competition, path2save)

    subprocess.run(f"unzip {zipf} -d {path2save}", shell=True)

    print("Deleting zip...")
    zipf.unlink(missing_ok=True)


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
