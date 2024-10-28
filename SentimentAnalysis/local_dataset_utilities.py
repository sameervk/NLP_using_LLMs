import os
import sys
import tarfile
import time

import numpy as np
import pandas as pd
import scipy.sparse._csr
from packaging import version
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import urllib
from pathlib import Path


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    if duration == 0:
        return
    progress_size = int(count * block_size)
    speed = progress_size / (1024.0**2 * duration)
    percent = count * block_size * 100.0 / total_size

    sys.stdout.write(
        f"\r{int(percent)}% | {progress_size / (1024.**2):.2f} MB "
        f"| {speed:.2f} MB/s | {duration:.2f} sec elapsed"
    )
    sys.stdout.flush()


def download_dataset(download_location: Path):
    source = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    target = download_location.joinpath("aclImdb_v1.tar.gz")

    if os.path.exists(target):
        os.remove(target)

    if not os.path.isdir(download_location.joinpath("aclImdb")) and not os.path.isfile(target):
        urllib.request.urlretrieve(source, target, reporthook)

    if not os.path.isdir(download_location.joinpath("aclImdb")):

        with tarfile.open(target, "r:gz") as tar:
            tar.extractall(path=download_location)


def load_dataset_into_to_dataframe(basepath: Path):

    labels = {"pos": 1, "neg": 0}

    df = pd.DataFrame()

    with tqdm(total=50000) as pbar:
        for s in ("test", "train"):
            for l in ("pos", "neg"):
                path = os.path.join(basepath, s, l)
                for file in sorted(os.listdir(path)):
                    with open(os.path.join(path, file), "r", encoding="utf-8") as infile:
                        txt = infile.read()

                    if version.parse(pd.__version__) >= version.parse("1.3.2"):
                        x = pd.DataFrame(
                            [[txt, labels[l]]], columns=["review", "sentiment"]
                        )
                        df = pd.concat([df, x], ignore_index=False)

                    else:
                        df = df.append([[txt, labels[l]]], ignore_index=True)
                    pbar.update()
    df.columns = ["text", "label"]

    return df


def partition_dataset(df: pd.DataFrame, basepath: Path):
    df_shuffled = df.sample(frac=1, random_state=1).reset_index()

    df_train = df_shuffled.iloc[:35_000]
    df_val = df_shuffled.iloc[35_000:40_000]
    df_test = df_shuffled.iloc[40_000:]

    df_train.to_csv(basepath.joinpath("train.csv"), index=False, encoding="utf-8")
    df_val.to_csv(basepath.joinpath("val.csv"), index=False, encoding="utf-8")
    df_test.to_csv("test.csv", index=False, encoding="utf-8")


# Creating a torch Dataset
class IMDBDataset(Dataset):

    def __init__(self, features: np.array, labels: np.array):

        # self.X = torch.tensor(df['text'].values)
        # This can only be done on numeric values

        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.uint64)
        # convert to tensor format

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):

        return len(self.y)

