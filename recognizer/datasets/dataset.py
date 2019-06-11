import abc
from pathlib import Path

from tensorflow.python import keras


class Dataset:
    """"Base dataset class"""

    @classmethod
    def data_path(cls):
        return Path(__file__).resolve().parents[2]

    @classmethod
    def raw_data_path(cls):
        return cls.data_path() / "data" / "raw"

    @classmethod
    def cache_data_path(cls):
        path = cls.data_path() / "data" / "cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def __init__(self, url: str, file_name: str, sha256: str):
        self.url = url
        self.file_name = file_name
        self.sha256 = sha256

        self.train_dataset = None
        self.test_dataset = None

        self._download()

    def _download(self):
        # https://www.tensorflow.org/alpha/tutorials/load_data/images
        download_path = keras.utils.get_file(
            origin=self.url,
            fname=self.file_name,
            file_hash=self.sha256,
            extract=True,
            cache_dir=Dataset.cache_data_path())
        download_path = Path(download_path)
        print(f"Download path: {download_path}")

        print("Processing data...")
        extracted_data_path = download_path.parent / self.file_name.split(".")[0]
        self._prepare(extracted_data_path)

    @abc.abstractmethod
    def _prepare(self, path):
        raise NotImplementedError("Dataset must override _prepare")


if __name__ == '__main__':
    import toml

    _raw_data_path = Dataset.raw_data_path()
    _metadata = toml.load(_raw_data_path / "emnist" / "metadata.toml")

    # print(f"Cache dir: {Dataset.cache_data_path()}")

    _download_path = keras.utils.get_file(
        origin=_metadata['url'],
        fname=_metadata['filename'],
        file_hash=_metadata['sha256'],
        extract=True,
        cache_dir=Dataset.cache_data_path())
    _download_path = Path(_download_path)
    print(_download_path.parent / _metadata['filename'].split(".")[0])
