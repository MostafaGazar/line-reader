#!/bin/bash
conda activate line-reader
python training/run_experiment.py --save '{"dataset":"EmnistDataset","network":"lenet","model":"CharacterModel","train_args":{"save_path":"recognizer/weights/character_model.h5","initial_learning_rate":0.01,"batch_size":256,"epochs":16,"checkpoints_path":null}}'
