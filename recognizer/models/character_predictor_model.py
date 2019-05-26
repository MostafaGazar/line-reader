from pathlib import Path
from typing import Callable

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.data import Dataset

from recognizer.models import Model
from recognizer.networks import NetworkInput


class CharacterModel(Model):

    def __init__(self, network: Callable[[NetworkInput], Model], save_path: Path):
        super().__init__(network, save_path)

        self.loss_object = keras.losses.CategoricalCrossentropy()

        initial_learning_rate = 0.01
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

    def train(self, checkpoints_path: Path, train_dataset: Dataset, valid_dataset: Dataset = None,
              batch_size: int = 256, epochs: int = 16):
        checkpoints_path.mkdir(parents=True, exist_ok=True)
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.network)
        manager = tf.train.CheckpointManager(ckpt, checkpoints_path, max_to_keep=3)
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print(f"Restored from {manager.latest_checkpoint}")
        else:
            print("Initializing from scratch.")

        # Start training the model.
        for epoch in range(1, epochs + 1):
            for images, labels in train_dataset:
                self._train_step(images, labels)

            for valid_images, valid_labels in valid_dataset:
                self._test_step(valid_images, valid_labels)

            ckpt.step.assign_add(1)
            if int(ckpt.step) % 10 == 0:
                save_path = manager.save()
                print(f"💾 Saved checkpoint for step {int(ckpt.step)}: {save_path}")

            print(f"Epoch {epoch}, "
                  f"Loss: {self.train_loss.result()}, Accuracy: {self.train_accuracy.result() * 100}, "
                  f"Test Loss: {self.test_loss.result()}, Test Accuracy: {self.test_accuracy.result() * 100}")

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
    _learning_rate_schedule = keras.optimizer.schedules.ExponentialDecay(
        .01,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    # optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate_schedule)
    print(_learning_rate_schedule)

    # network = lenet(NetworkInput(input_shape=input_shape, mean=emnist.mean, std=emnist.std, number_of_classes=emnist.number_of_classes))
