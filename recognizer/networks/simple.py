from tensorflow.python.keras import Model as KerasModel
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential

from recognizer.networks import NetworkInput
from recognizer.utils import norm


# Shape: https://www.tensorflow.org/api_docs/python/tf/shape
def simple(network_input: NetworkInput) -> KerasModel:
    model = Sequential([
        layers.Lambda(lambda x: norm(x, network_input.mean, network_input.std), input_shape=network_input.input_shape, output_shape=network_input.input_shape),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(units=network_input.number_of_classes, activation='softmax')
    ])

    return model
