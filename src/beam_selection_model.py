from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Dropout

import numpy as np


def build_resblock(input_shape, nChannels):
    input = Input(shape=input_shape)
    x = Conv2D(nChannels, (3, 3), padding="same", activation="relu")(input)
    x = Conv2D(nChannels, (3, 3), padding="same", activation=None)(x)
    x = input + x
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)
    return Model(inputs=input, outputs=x)


def build_model(input_shape, nClasses):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (7, 7), padding="same", activation="relu"))
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(build_resblock(model.layers[-1].output_shape[1:], 32))
    model.add(build_resblock(model.layers[-1].output_shape[1:], 32))
    model.add(build_resblock(model.layers[-1].output_shape[1:], 32))
    model.add(Flatten())
    model.add(Dense(2 * nClasses, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=nClasses))
    model.add(Dropout(0.5))
    model.add(Softmax())

    return model


if __name__ == "__main__":
    model = build_model(input_shape=(28, 210, 1), nClasses=64)
    input = np.zeros((32, 28, 210, 1))
    print(model(input).shape)
