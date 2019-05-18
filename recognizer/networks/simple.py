from tensorflow.python import Shape
from tensorflow.python.keras import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers


# Shape: https://www.tensorflow.org/api_docs/python/tf/shape
def simple(input_shape: Shape, number_of_classes: int) -> Model:
    model = Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(units=number_of_classes, activation='softmax')
    ])

    return model
