import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import os


def custom_f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = TP / (Positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = TP / (Pred_Positives + K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def plot_history(csv_file, pictures_folder):
    df = pd.read_csv(csv_file)

    if type(pictures_folder) == pathlib.PosixPath:
        if not pictures_folder.is_dir():
            pictures_folder.mkdir(parents=True, exist_ok=True)
    else:
        if not os.path.isdir(pictures_folder):
            os.mkdir(pictures_folder)

    # summarize history for accuracy
    plt.plot(df["epoch"].values, df["accuracy"].values)
    plt.plot(df["epoch"].values, df["val_accuracy"].values)
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    if type(pictures_folder) == pathlib.PosixPath:
        plt.savefig(pictures_folder / "accuracy.png")
    else:
        plt.savefig(os.path.join(pictures_folder, "accuracy.png"))
    plt.close()
    plt.close()
    # summarize history for loss
    plt.plot(df["epoch"].values, df["loss"].values)
    plt.plot(df["epoch"].values, df["val_loss"].values)
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    if type(pictures_folder) == pathlib.PosixPath:
        plt.savefig(pictures_folder / "loss.png")
    else:
        plt.savefig(os.path.join(pictures_folder, "loss.png"))
    plt.close()


def plot_test_results(csv_file, pictures_folder):
    df = pd.read_csv(csv_file)

    if type(pictures_folder) == pathlib.PosixPath:
        if not pictures_folder.is_dir():
            pictures_folder.mkdir(parents=True, exist_ok=True)
    else:
        if not os.path.isdir(pictures_folder):
            os.mkdir(pictures_folder)

    # summarize history for accuracy
    plt.plot(df.keys().tolist()[1:], df.values[0, 1:], marker="o")
    plt.title("Top-K accuracy")
    plt.ylabel("Percentage")
    plt.xlabel("Top-K")
    if type(pictures_folder) == pathlib.PosixPath:
        plt.savefig(pictures_folder / "topk.png")
    else:
        plt.savefig(os.path.join(pictures_folder, "topk.png"))
    plt.close()
    plt.close()


if __name__ == "__main__":
    import argparse
    import json
    import attrdict
    import numpy as np
    from dataGenerator import DataGenerator

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_file", "-c", help="Path to config file", type=str, required=True
    )
    args = parser.parse_args()

    with open(args.cfg_file, "r") as cfg_file:
        cfg = attrdict.AttrDict(json.load(cfg_file))

    datagen = DataGenerator(
        mode="train", cfg=cfg.DataGenerator, generateDataStructure=False
    )

    x, y = datagen[0]
    output = np.random.randn(*y.shape).astype(np.float32)

    print(custom_f1(y, output))
