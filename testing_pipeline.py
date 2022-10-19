import pandas as pd
import numpy as np
import argparse
import attrdict
import json
import os
from termcolor import colored, cprint
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.metrics import TopKCategoricalAccuracy

from src.beam_selection_model import build_model
from src.datagenerator import DataGenerator
from src.utils import plot_history, plot_test_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_file", "-c", help="Path to config file", type=str, required=True
    )
    args = parser.parse_args()

    with open(args.cfg_file, "r") as cfg_file:
        cfg = attrdict.AttrDict(json.load(cfg_file))

    print(
        colored("# Declaring Data generators...", "green", attrs=["reverse", "blink"])
    )
    test_datagen = DataGenerator(mode="test", cfg=cfg.DataGenerator)

    print(colored("# Declaring sequence model...", "green", attrs=["reverse", "blink"]))
    model = build_model(
        input_shape=cfg.DataGenerator.posMatShape,
        nClasses=cfg.DataGenerator.nTx * cfg.DataGenerator.nRx,
    )

    if cfg.Optimizers.choise.lower() == "adam":
        print(
            colored(
                "# Setting up Adam optimizer...", "green", attrs=["reverse", "blink"]
            )
        )
        opt = Adam(
            learning_rate=cfg.Optimizers.Adam.learning_rate,
            beta_1=cfg.Optimizers.Adam.beta_1,
            beta_2=cfg.Optimizers.Adam.beta_2,
            epsilon=cfg.Optimizers.Adam.epsilon,
            amsgrad=cfg.Optimizers.Adam.amsgrad,
            name=cfg.Optimizers.Adam.name,
        )
    elif cfg.Optimizers.choise.lower() == "rmsprop":
        print(
            colored(
                "# Setting up RMSprop optimizer...", "green", attrs=["reverse", "blink"]
            )
        )
        opt = RMSprop(
            learning_rate=cfg.Optimizers.RMSprop.learning_rate,
            rho=cfg.Optimizers.RMSprop.rho,
            momentum=cfg.Optimizers.RMSprop.momentum,
            epsilon=cfg.Optimizers.RMSprop.epsilon,
            centered=cfg.Optimizers.RMSprop.centered,
            name=cfg.Optimizers.RMSprop.name,
        )
    print(colored("# Compiling the model...", "green", attrs=["reverse", "blink"]))
    model.compile(
        optimizer=opt,
        loss=CategoricalCrossentropy(),
        metrics=[
            TopKCategoricalAccuracy(k=i, name="top{}".format(i)) for i in range(1, 11)
        ],
    )

    model.load_weights(
        "./models/model_epochs-{}_opt-{}".format(
            cfg.epochs, cfg.Optimizers.choise.lower()
        )
    )

    print(
        colored(
            "# Starting the training process...", "green", attrs=["reverse", "blink"]
        )
    )

    filename = "logs/test_log_epochs-{}_opt-{}.csv".format(
        cfg.epochs, cfg.Optimizers.choise.lower()
    )

    scores = model.evaluate(
        x=test_datagen,
        batch_size=cfg.DataGenerator.batchsize,
        steps=len(test_datagen),
    )

    filename = "logs/test_log_epochs-{}_opt-{}.csv".format(
        cfg.epochs, cfg.Optimizers.choise.lower()
    )
    summary = dict(zip(model.metrics_names, scores))
    summary = pd.DataFrame(summary, index=[0])
    summary.to_csv(filename, index=False)
