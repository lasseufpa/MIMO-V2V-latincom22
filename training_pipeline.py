import numpy as np
import argparse
import attrdict
import json
import os
from termcolor import colored, cprint
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

from src.beam_selection_model import build_model
from src.datagenerator import DataGenerator
from src.utils import plot_history

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
    train_datagen = DataGenerator(
        mode="train", cfg=cfg.DataGenerator
    )
    val_datagen = DataGenerator(
        mode="val", cfg=cfg.DataGenerator
    )

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
    model.compile(optimizer=opt, loss=CategoricalCrossentropy(), metrics=["accuracy"])

    print(
        colored(
            "# Starting the training process...", "green", attrs=["reverse", "blink"]
        )
    )
    if not os.path.isdir("logs"):
        os.mkdir("logs")
    filename = "logs/log_epochs-{}_opt-{}.csv".format(
        cfg.epochs, cfg.Optimizers.choise.lower()
    )
    history_logger = CSVLogger(filename, separator=",", append=True)
    if not os.path.isdir("models"):
        os.mkdir("models")
    model_checkpoint_callback = ModelCheckpoint(
        filepath="models/model_epochs-{}_opt-{}".format(
            cfg.epochs, cfg.Optimizers.choise.lower()
        ),
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    )
    model.fit_generator(
        generator=train_datagen,
        validation_data=val_datagen,
        epochs=cfg.epochs,
        steps_per_epoch=len(train_datagen),
        callbacks=[history_logger, model_checkpoint_callback],
    )

    print(
        colored(
            "# Ploting pictures...", "green", attrs=["reverse", "blink"]
        )
    )
    pictures_folder="logs/log_epochs-{}_opt-{}/".format(
        cfg.epochs, cfg.Optimizers.choise.lower()
    )
    plot_history(filename,pictures_folder)