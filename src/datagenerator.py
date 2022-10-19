import numpy as np
import pathlib
import json
import argparse
import attrdict
from tqdm import tqdm
import pandas as pd
from tensorflow import keras
from tensorflow.keras.utils import to_categorical


class DataGenerator(keras.utils.Sequence):
    def __init__(self, mode, cfg) -> None:
        self.datasetPath = cfg.datasetPath
        self.nTx = cfg.nTx
        self.nRx = cfg.nRx
        self.batchsize = cfg.batchsize
        np.random.seed(cfg.seed)
        if mode == "train":
            self.samplesCSV = pd.read_csv(
                pathlib.Path(self.datasetPath).resolve() / "train.csv"
            )
        elif mode == "val":
            self.samplesCSV = pd.read_csv(
                pathlib.Path(self.datasetPath).resolve() / "val.csv"
            )
        elif mode == "test":
            self.samplesCSV = pd.read_csv(
                pathlib.Path(self.datasetPath).resolve() / "test.csv"
            )
        else:
            pass

    def on_epoch_end(self) -> None:
        self.samplesCSV = self.samplesCSV.sample(frac=1)

    def __len__(self) -> int:
        return int(np.floor(len(self.samplesCSV) / self.batchsize))

    def __getitem__(self, index) -> tuple:
        samples = self.samplesCSV.iloc[
            index * self.batchsize : (index + 1) * self.batchsize
        ]

        X = []
        Y = []
        for i in range(len(samples)):
            X.append(np.load(samples.iloc[i]["img_path"])["positionmatrix"])
            Y.append(samples.iloc[i]["label"])

        return (np.array(X, dtype=float), to_categorical(Y, self.nTx * self.nRx))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_file", "-c", help="Path to config file", type=str, required=True
    )
    args = parser.parse_args()

    with open(args.cfg_file, "r") as cfg_file:
        cfg = attrdict.AttrDict(json.load(cfg_file))

    datagen = DataGenerator(mode="train", cfg=cfg.DataGenerator)

    for i in range(len(datagen)):
        x, y = datagen[i]
        print(x.shape, y.shape)
