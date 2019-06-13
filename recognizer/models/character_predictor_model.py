from pathlib import Path
from typing import Callable

import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.python.keras import Model as KerasModel

from recognizer.models.base import Model
from recognizer.networks import NetworkInput


class CharacterModel(Model):

    def __init__(self, network: Callable[[NetworkInput], KerasModel], save_path: Path, initial_learning_rate: float = 0.01):
        super().__init__(network, save_path)

        self.loss_object = keras.losses.CategoricalCrossentropy()

        learning_rate_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True)
        self.optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate_schedule)

        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.train_accuracy = keras.metrics.CategoricalAccuracy(name='train_accuracy')

        self.test_loss = keras.metrics.Mean(name='test_loss')
        self.test_accuracy = keras.metrics.CategoricalAccuracy(name='test_accuracy')

    def train(self, train_dataset: Dataset, valid_dataset: Dataset = None,
              batch_size: int = 256, epochs: int = 16, checkpoints_path: Path = None):
        print("Training model...")

        ckpt = None
        manager = None
        if checkpoints_path is not None:
            checkpoints_path.mkdir(parents=True, exist_ok=True)
            ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.network)
            manager = tf.train.CheckpointManager(ckpt, checkpoints_path, max_to_keep=3)
            ckpt.restore(manager.latest_checkpoint)
            if manager.latest_checkpoint:
                print(f"Restored from {manager.latest_checkpoint}")
            else:
                print("Initializing from scratch.")

        # Batch the datasets
        train_dataset = train_dataset.shuffle(1024).batch(batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        valid_dataset = valid_dataset.batch(batch_size)

        # Start training the model.
        for epoch in range(1, epochs + 1):
            for images, labels in train_dataset:
                self._train_step(images, labels)

            for valid_images, valid_labels in valid_dataset:
                self._test_step(valid_images, valid_labels)

            if checkpoints_path is not None:
                ckpt.step.assign_add(1)
                if int(ckpt.step) % 10 == 0:
                    save_path = manager.save()
                    print(f"💾 Saved checkpoint for step {int(ckpt.step)}: {save_path}")

            print(f"Epoch {epoch}, "
                  f"Loss: {self.train_loss.result()}, Accuracy: {self.train_accuracy.result() * 100}, "
                  f"Valid Loss: {self.test_loss.result()}, Valid Accuracy: {self.test_accuracy.result() * 100}")

        # Save the model.
        self.network.trainable = False
        self.network.save(self.save_path)

    @tf.function
    def _train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.network(images)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def _test_step(self, images, labels):
        predictions = self.network(images)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)


if __name__ == '__main__':
    from recognizer.datasets import EmnistDataset
    from recognizer.networks import lenet

    _emnist = EmnistDataset()
    _train_dataset = _emnist.train_dataset
    _valid_dataset = _emnist.test_dataset

    (x_train, y_train), = _train_dataset.take(1)
    input_shape = tuple(x_train.shape) # Use x_train[0] when batched
    print(f"x shape: {x_train.shape}, model input shape: {input_shape}")

    _network = lenet(NetworkInput(input_shape=input_shape, mean=_emnist.mean, std=_emnist.std,
                                  number_of_classes=_emnist.number_of_classes))
    _model = CharacterModel(network=_network, save_path=Path("../recognizer/weights/character_model.h5"))

    _model.train(checkpoints_path=Path("../recognizer/ckpts/character_model"), train_dataset=_train_dataset,
                 valid_dataset=_valid_dataset)
