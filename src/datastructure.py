import numpy as np
import pandas as pd
import argparse
import attrdict
import json
import pathlib


def create_data_structure(cfg):
    input_data = np.load(cfg.input_data)
    output_data = np.load(cfg.output_data)
    positionmatrix = input_data[input_data.files[0]]
    rays = output_data[output_data.files[0]]
    del input_data
    del output_data

    root = pathlib.Path("./data")

    if not root.is_dir():
        root.mkdir()

    positionmatrix = np.expand_dims(positionmatrix, axis=-1)

    summary = []

    for subfolder in ["{}".format(i) for i in range(cfg.nTx * cfg.nRx)]:
        if not (root / subfolder).is_dir():
            (root / subfolder).mkdir()

    for i in range(positionmatrix.shape[0]):
        image_path = root.resolve() / "{}".format(int(rays[i])) / "{}.npz".format(i)
        np.savez(image_path, positionmatrix=positionmatrix[i])
        summary.append([i, image_path, int(rays[i])])

    df = pd.DataFrame(columns=["scene", "img_path", "label"], data=summary)
    df.to_csv(root / "summary.csv", index=False)
    df = df.sample(frac=1)
    train, val, test = np.split(
        df, [int(len(df) * cfg.splits[0]), int(len(df) * sum(cfg.splits[:2]))]
    )
    train.to_csv(root / "train.csv", index=False)
    val.to_csv(root / "val.csv", index=False)
    test.to_csv(root / "test.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_file", "-c", help="Path to config file", type=str, required=True
    )
    args = parser.parse_args()

    with open(args.cfg_file, "r") as cfg_file:
        cfg = attrdict.AttrDict(json.load(cfg_file))

    create_data_structure(cfg=cfg)
