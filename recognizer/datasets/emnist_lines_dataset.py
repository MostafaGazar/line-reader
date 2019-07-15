import itertools
import re
import string

import nltk

from recognizer.datasets import Dataset

NLTK_DATA_DIRNAME = Dataset.raw_data_path() / 'nltk'


class EmnistLinesDataset(Dataset):

    # def __init__(self):
    #     self.emnist = EmnistDataset()
    #
    #     raw_data_path = Dataset.raw_data_path()
    #     metadata = toml.load(raw_data_path / "emnist" / "metadata.toml")
    #
    #     super().__init__(url=metadata["url"], file_name=metadata["filename"], sha256=metadata["sha256"])

    def _prepare(self, path):
        pass

    def _generate_data(self):
        pass

    @staticmethod
    def _brown_text():
        """Return a single string with the Brown corpus with all punctuation stripped."""
        sents = EmnistLinesDataset._load_nltk_brown_corpus()
        text = ' '.join(itertools.chain.from_iterable(sents))
        text = text.translate({ord(c): None for c in string.punctuation})
        text = re.sub('  +', ' ', text)

        return text

    @staticmethod
    def _load_nltk_brown_corpus():
        """Load the Brown corpus using the NLTK library."""
        nltk.data.path.append(NLTK_DATA_DIRNAME)
        try:
            nltk.corpus.brown.sents()
        except LookupError:
            NLTK_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
            nltk.download('brown', download_dir=NLTK_DATA_DIRNAME)

        return nltk.corpus.brown.sents()


if __name__ == '__main__':
    _text = EmnistLinesDataset._brown_text()
    print(_text)

    # _dataset = EmnistLinesDataset()
    # print(f"Number of classes: {_dataset.number_of_classes}")
    # print(f"mean: {_dataset.mean}, std: {_dataset.std}")
    #
    # (_image, _label), = _dataset.train_dataset.take(1)
    # # Convert the label tensor to numpy array and then get its Python scalar value.
    # print(f"Image shape: {_image.shape}, label: {_label}")
