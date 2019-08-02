import tensorflow as tf

from tensorflow.python.keras import Model as KerasModel
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential

from recognizer.networks import NetworkInput
from recognizer.utils import norm


# Shape: https://www.tensorflow.org/api_docs/python/tf/shape
def lenet(network_input: NetworkInput) -> KerasModel:
    model = Sequential()

    input_shape = network_input.input_shape
    if len(network_input.input_shape) < 3:
        model.add(layers.Lambda(lambda x: tf.expand_dims(x, -1), input_shape=input_shape))
        input_shape = (input_shape[0], input_shape[1], 1)

    if network_input.mean is not None and network_input.std is not None:
        model.add(layers.Lambda(lambda x: norm(x, network_input.mean, network_input.std), input_shape=input_shape))

    model.add(layers.Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(network_input.number_of_classes, activation='softmax'))

    return model
