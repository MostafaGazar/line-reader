import abc
from pathlib import Path
from typing import Callable

from tensorflow.python.data import Dataset
from tensorflow.python.keras import Model

from recognizer.networks import NetworkInput


class Model:

    def __init__(self, network: Callable[[NetworkInput], Model], save_path: Path):
        self.network = network
        self.save_path = save_path

    @abc.abstractmethod
    def train(self, train_dataset: Dataset, checkpoints_path: Path, valid_dataset: Dataset = None,
              batch_size: int = 256, epochs: int = 16):
        pass
