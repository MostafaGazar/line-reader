import abc
from pathlib import Path
from typing import Callable

from tensorflow.python.data import Dataset
from tensorflow.python.keras import Model as KerasModel

from recognizer.networks import NetworkInput


class Model:

    def __init__(self, network: Callable[[NetworkInput], KerasModel], save_path: Path):
        self.network = network
        self.save_path = save_path

    @abc.abstractmethod
    def train(self, train_dataset: Dataset, valid_dataset: Dataset = None,
              batch_size: int = 256, epochs: int = 16, checkpoints_path: Path = None):
        raise NotImplementedError("Dataset must override _prepare")
