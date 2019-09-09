from tensorflow.python.keras import Model as KerasModel
from tensorflow.python.keras.layers import *

from recognizer.networks import NetworkInput


def unet(network_input: NetworkInput) -> KerasModel:
#     """
#     :return:  model -- a model that has been defined, but not yet compiled.
#                       The model is an implementation of the Unet paper
#                       (https://arxiv.org/pdf/1505.04597.pdf) and comes
#                       from this repo https://github.com/zhixuhao/unet. It has
#                       been modified to keep up with API changes in keras 2.
#     """
#     inputs = Input(network_input.input_shape)
    
#     conv1 = Conv2D(filters=64,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(inputs)
#     conv1 = Conv2D(filters=64,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
#     conv2 = Conv2D(filters=128,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(pool1)
#     conv2 = Conv2D(filters=128,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
#     conv3 = Conv2D(filters=256,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(pool2)
#     conv3 = Conv2D(filters=256,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
#     conv4 = Conv2D(filters=512,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(pool3)
#     conv4 = Conv2D(filters=512,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv4)
#     drop4 = Dropout(0.5)(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
#     conv5 = Conv2D(filters=1024,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(pool4)
#     conv5 = Conv2D(filters=1024,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv5)
#     drop5 = Dropout(0.5)(conv5)
    
#     up6 = UpSampling2D(size=(2, 2))(drop5)
#     up6 = Conv2D(filters=512,
#                  kernel_size=2,
#                  activation='relu',
#                  padding='same',
#                  kernel_initializer='he_normal')(up6)
#     merge6 = Concatenate(axis=3)([drop4, up6])
#     conv6 = Conv2D(filters=512,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(merge6)
#     conv6 = Conv2D(filters=512,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv6)
    
#     up7 = UpSampling2D(size=(2, 2))(conv6)
#     up7 = Conv2D(filters=256,
#                  kernel_size=2,
#                  activation='relu',
#                  padding='same',
#                  kernel_initializer='he_normal')(up7)
#     merge7 = Concatenate(axis=3)([conv3, up7])
#     conv7 = Conv2D(filters=256,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(merge7)
#     conv7 = Conv2D(filters=256,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv7)
    
#     up8 = UpSampling2D(size=(2, 2))(conv7)
#     up8 = Conv2D(filters=128,
#                  kernel_size=2,
#                  activation='relu',
#                  padding='same',
#                  kernel_initializer='he_normal')(up8)
#     merge8 = Concatenate(axis=3)([conv2, up8])
#     conv8 = Conv2D(filters=128,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(merge8)
#     conv8 = Conv2D(filters=128,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv8)
    
#     up9 = UpSampling2D(size=(2, 2))(conv8)
#     up9 = Conv2D(filters=64,
#                  kernel_size=2,
#                  activation='relu',
#                  padding='same',
#                  kernel_initializer='he_normal')(up9)
#     merge9 = Concatenate(axis=3)([conv1, up9])
#     conv9 = Conv2D(filters=64,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(merge9)
#     conv9 = Conv2D(filters=64,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv9)
#     conv9 = Conv2D(filters=2,
#                    kernel_size=3,
#                    activation='relu',
#                    padding='same',
#                    kernel_initializer='he_normal')(conv9)
#     conv10 = Conv2D(filters=network_input.number_of_classes, kernel_size=1, activation='sigmoid')(conv9)
    
#     model = KerasModel(inputs=inputs, outputs=conv10)
#     model.model_name = "unet"
    
#     return model

    # Based on https://github.com/divamgupta/image-segmentation-keras/blob/master/keras_segmentation/models/unet.py#L19
    inputs = Input(network_input.input_shape)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D((2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D((2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

    outputs = Conv2D(network_input.number_of_classes, (1, 1), padding='same')(conv5)

    model = KerasModel(inputs=inputs, outputs=outputs)
    model.model_name = "unet_mini"

    return model


if __name__ == '__main__':
    input_shape = (256, 256, 1)
    number_of_classes = 3

    model = unet(NetworkInput(input_shape=input_shape, number_of_classes=number_of_classes))

    model.summary()
