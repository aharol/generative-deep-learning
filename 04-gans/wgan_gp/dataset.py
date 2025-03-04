import zipfile
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
from tempfile import (
        NamedTemporaryFile,
        TemporaryDirectory
)
from mlx.data.datasets.common import (
        CACHE_DIR,
        ensure_exists,
        urlretrieve_with_progress
)
from PIL import Image
import mlx.data as dx

data_splits = {
    "train": [],
    "test": [],
    "eval": []
}


def _load_celeba_wrapper(root: Optional[str|Path]=None, split: str="train", quiet: bool=False) -> List[Dict]:

    base_url = "https://www.kaggle.com/api/v1/datasets/download/jessicali9530/celeba-dataset"

    if root is None:
        root = CACHE_DIR / "celeba_dataset"
    else:
        root = Path(root)
    ensure_exists(root)

    def download():

        temp_file = NamedTemporaryFile()
        urlretrieve_with_progress(src=base_url, dst=temp_file.name, quiet=quiet)

        temp_dir = TemporaryDirectory()
        with zipfile.ZipFile(temp_file.name, 'r') as zf:
            zf.extractall(temp_dir.name)

            image_dir = Path(temp_dir.name) / "img_align_celeba/img_align_celeba"
            attrs = pd.read_csv(Path(temp_dir.name) / "list_attr_celeba.csv").set_index("image_id")
            bbox = pd.read_csv(Path(temp_dir.name) / "list_bbox_celeba.csv").set_index("image_id")
            eval_partition = pd.read_csv(Path(temp_dir.name) / "list_eval_partition.csv").set_index("image_id")
            landmarks_align = pd.read_csv(Path(temp_dir.name) / "list_landmarks_align_celeba.csv").set_index("image_id")

            for fname in image_dir.glob("*.jpg"):
                idx = fname.name
                sample = {
                    "image": np.array(Image.open(fname)),
                    "bbox": bbox.loc[idx].to_numpy(),
                    "attrs": attrs.loc[idx].to_numpy(),
                    "land_marks": landmarks_align.loc[idx].to_numpy()
                }
                if eval_partition.loc[idx].item() == 0:
                    data_splits["train"].append(sample)
                elif eval_partition.loc[idx].item() == 1:
                    data_splits["eval"].append(sample)
                else:
                    data_splits["test"].append(sample)

        for ds in data_splits:
            if len(data_splits[ds]) == 0:
                print(f"Warning: data split '{ds}' is empty!")
            with (root / f"celeba_{ds}.pkl").open("wb") as f:
                pickle.dump(data_splits[ds], f)


    if split not in data_splits:
        raise ValueError(f"The '{split}' split is irrelevant. Any of '{list(data_splits)}' is allowed.")

    if not (root / f"celeba_{split}.pkl").is_file():
        download()

    with (root / f"celeba_{split}.pkl").open("rb") as f:
        return dx.buffer_from_vector(pickle.load(f))


def load_celeba(root=None, split=None, quiet=False):
    """Load a buffer with the CelebA dataset.

    If the data doesn't exist download it and save it for the next time.
    Because the dataset is relatively small (40_000 samples), it is prepared
    as a pickled Numpy array.

    The dataset has no labels, and loaded as RGB images of size 64x64

    Args:
        root (Path or str, optional): The directory to load/save the data.
            Default: `~/.cache/mlx.data/celeba_{train|eval|test}.pkl`.
        split (str, optional): Which data partition to load.
            One of "train", "eval" or "test". Default: "train".
        quiet (bool, optional): Download quietly (Default: False).
    """

    if split is None:
        split = "train"

    return _load_celeba_wrapper(root, split, quiet)
