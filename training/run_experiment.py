import argparse
import json
from importlib import import_module

from recognizer.networks import NetworkInput


def run_experiment(experiment_config: dict):
    """
    {
      "dataset": "EmnistDataset",
      "network": "lenet",
      "model": "CharacterModel",
      "train_args": {
        "save_path": "recognizer/weights/character_model.h5",
        "initial_learning_rate": 0.01,
        "batch_size": 256,
        "epochs": 16,
        "checkpoints_path": null
      }
    }
    """
    datasets_module = import_module("recognizer.datasets")
    networks_module = import_module("recognizer.networks")
    models_module = import_module("recognizer.models")

    dataset_class = getattr(datasets_module, experiment_config['dataset'])
    dataset = dataset_class(**experiment_config.get('dataset_args', {}))

    (x_train, y_train), = dataset.train_dataset.take(1)
    input_shape = tuple(x_train.shape)  # Use x_train[0] when batched
    print(f"x shape: {x_train.shape}, model input shape: {input_shape}")
    network_input = NetworkInput(input_shape=input_shape, mean=dataset.mean, std=dataset.std,
                                 number_of_classes=dataset.number_of_classes)

    network_class = getattr(networks_module, experiment_config['network'])
    network = network_class(network_input)

    model_class = getattr(models_module, experiment_config['model'])
    model = model_class(
        network=network,
        save_path=experiment_config['train_args']['save_path'],
        initial_learning_rate=experiment_config['train_args']['initial_learning_rate'])
    model.train(
        train_dataset=dataset.train_dataset,
        valid_dataset=dataset.test_dataset,
        batch_size=experiment_config['train_args']['batch_size'],
        epochs=experiment_config['train_args']['epochs'],
        checkpoints_path=experiment_config['train_args']['checkpoints_path'])


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "experiment_config",
        type=str,
        help="{\"dataset\":\"EmnistDataset\",\"network\":\"lenet\",\"model\":\"CharacterModel\",\"train_args\":{\"save_path\":\"recognizer/weights/character_model.h5\",\"initial_learning_rate\":0.01,\"batch_size\":256,\"epochs\":16,\"checkpoints_path\":null}}"
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """Run experiment"""
    _args = _parse_args()

    _experiment_config = json.loads(_args.experiment_config)
    run_experiment(_experiment_config)

    # _experiment_config = json.loads("{\"dataset\":\"EmnistDataset\",\"network\":\"lenet\",\"model\":\"CharacterModel\",\"train_args\":{\"save_path\":\"recognizer/weights/character_model.h5\",\"initial_learning_rate\":0.01,\"batch_size\":256,\"epochs\":16,\"checkpoints_path\":null}}")
    # run_experiment(_experiment_config)
