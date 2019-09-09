from typing import List
from tensorflow.python.keras import Model as KerasModel
from tensorflow.python.keras.layers import *

from recognizer.networks import NetworkInput


# Based on https://github.com/full-stack-deep-learning/fsdl-text-recognizer-project/blob/master/lab6_sln/text_recognizer/networks/fcn.py
def _residual_conv_block(input_layer: Layer,
                        kernel_sizes: List[int],
                        num_filters: List[int],
                        dilation_rates: List[int],
                        activation: str) -> Layer:
    """Function to instantiate a Residual convolutional block."""
    padding = 'same'
    x = Conv2D(num_filters[0],
               kernel_size=kernel_sizes[0],
               dilation_rate=dilation_rates[0],
               padding=padding,
               activation=activation)(input_layer)
    x = Conv2D(num_filters[1], kernel_size=kernel_sizes[1], dilation_rate=dilation_rates[1], padding=padding)(x)
    y = Conv2D(num_filters[1], kernel_size=1, dilation_rate=1, padding=padding)(input_layer)
    x = Add()([x, y])
    x = Activation(activation)(x)
    return x


def fcn(network_input: NetworkInput) -> KerasModel:
    """Function to instantiate a fully convolutional residual network for line detection."""
    num_filters = [16] * 14
    kernel_sizes = [7] * 14
    dilation_rates = [3] * 4 + [7] * 10

    input_image = Input(network_input.input_shape)

    for i in range(0, len(num_filters), 2):
        model_layer = _residual_conv_block(input_layer=input_image,
                                          kernel_sizes=kernel_sizes[i:i+2],
                                          num_filters=num_filters[i:i+2],
                                          dilation_rates=dilation_rates[i:i+2],
                                          activation='relu')
    output = Conv2D(network_input.number_of_classes, kernel_size=1, dilation_rate=1, padding='same', activation='softmax')(model_layer)

    model = KerasModel(inputs=input_image, outputs=output)
    return model
