from tensorflow.python.keras import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential

from recognizer.networks import NetworkInput
from recognizer.utils import norm


# Shape: https://www.tensorflow.org/api_docs/python/tf/shape
def lenet(network_input: NetworkInput) -> Model:
    model = Sequential()

    model.add(layers.Lambda(lambda x: norm(x, network_input.mean, network_input.std), input_shape=network_input.input_shape, output_shape=network_input.input_shape))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(network_input.number_of_classes, activation='softmax'))

    return model
