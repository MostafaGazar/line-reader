from recognizer.datasets import Dataset

import tensorflow as tf
import numpy as np
import toml
import json
from scipy.io import loadmat


class EmnistDataset(Dataset):
    """
    "The EMNIST dataset is a set of handwritten character digits derived from the NIST Special Database 19
    and converted to a 28x28 pixel image format and dataset structure that directly matches the MNIST dataset."
    From https://www.nist.gov/itl/iad/image-group/emnist-dataset

    The data split we will use is
    EMNIST ByClass: 814,255 characters. 62 unbalanced classes.
    """

    def __init__(self):
        raw_data_path = Dataset.raw_data_path()
        metadata = toml.load(raw_data_path / "emnist" / "metadata.toml")

        super().__init__(url=metadata["url"], file_name=metadata["filename"], sha256=metadata["sha256"])

    def _prepare(self, path):
        self.mapping = self._load_labels_mapping()

        data = loadmat(path / "emnist-byclass.mat")

        # load training dataset
        x_train = data["dataset"][0][0][0][0][0][0].astype(np.float32).reshape(-1, 28, 28, order="A")
        y_train = data["dataset"][0][0][0][0][0][1]
        print("Balancing train dataset...")
        x_train, y_train = self._sample_to_balance(x_train, y_train)

        # load test dataset
        x_test = data["dataset"][0][0][1][0][0][0].astype(np.float32).reshape(-1, 28, 28, order="A")
        y_test = data["dataset"][0][0][1][0][0][1]

        self.train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    @staticmethod
    def _load_labels_mapping():
        with open("emnist_essentials.json") as json_file:
            mapping = json.load(json_file)["mapping"]
            mapping = {m[0]: m[1] for m in mapping}

        return mapping

    @staticmethod
    def _sample_to_balance(x, y):
        """Because the dataset is not balanced, we take at most the mean number of instances per class."""
        num_to_sample = int(np.bincount(y.flatten()).mean())
        print(f"Target max number of images per class: {num_to_sample}")

        all_sampled_indices = []
        for label in np.unique(y.flatten()):
            indices = np.where(y == label)[0]
            sampled_indices = np.unique(np.random.choice(indices, num_to_sample))
            all_sampled_indices.append(sampled_indices)

        ind = np.concatenate(all_sampled_indices)
        x_sampled = x[ind]
        y_sampled = y[ind]

        return x_sampled, y_sampled


if __name__ == '__main__':
    _dataset = EmnistDataset()

    (_image, _label), = _dataset.train_dataset.take(1)
    # Convert the label tensor to numpy array and then get its Python scalar value.
    print(f"Image shape: {_image.shape}, label: {_label} - {_dataset.mapping[_label.numpy().item()]}")
