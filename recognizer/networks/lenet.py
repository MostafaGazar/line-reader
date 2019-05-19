from tensorflow.python import Shape
from tensorflow.python.keras import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers


# Shape: https://www.tensorflow.org/api_docs/python/tf/shape
def lenet5(input_shape: Shape, number_of_classes: int) -> Model:
    model = Sequential()

#     model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=input_shape))  # (32, 32, 1)
#     model.add(layers.AveragePooling2D())

#     model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
#     model.add(layers.AveragePooling2D())

#     model.add(layers.Flatten())

#     model.add(layers.Dense(units=120, activation='relu'))
#     model.add(layers.Dense(units=84, activation='relu'))
#     model.add(layers.Dense(units=number_of_classes, activation='softmax'))

    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(number_of_classes, activation='softmax'))

    return model
