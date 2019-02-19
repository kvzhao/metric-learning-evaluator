"""
    Main function for the offline evaluation.
    This program load extracted features and analyze results.

    Assumption:
      1. Folder in the structure
        folder
        ├── embeddings.npy
        ├── filename_strings.npy
        └── labels.npy

    TODO @kv:
      This program would handle source data from folder (with numpy arrays)
      tfrecord or dataset backbone.

    NOTE: for developing
    Quickly build up this program.
        1. Json wrapper
        2. Dataset reader
        3. Metric functions
"""

import os
import sys
import yaml
import numpy as np

from pprint import pprint

from evaluator.evaluator_builder import EvaluatorBuilder

def main(args):

    config = args.config
    data_type = args.data_type
    data_dir = args.data_dir

    if not data_dir:
        raise ValueError('data_dir must be assigned!')

    # open file
    evaluator = EvaluatorBuilder(args.config)

    if data_type == 'folder':
        path_embeddings = data_dir + '/embeddings.npy'
        path_filenames = data_dir + '/filename_strings.npy'
        path_labels = data_dir + '/labels.npy'

        embeddings = np.load(path_embeddings)
        filenames = np.load(path_filenames)
        labels = np.load(path_labels)

    # Add datum through loop

    for feat, label, fn in zip(embeddings, labels, filenames):
        evaluator.add_image_id_and_embedding(fn, label, feat)
    total_results = evaluator.evaluate()

    pprint (total_results)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Offline Metric Learning Evaluation')

    parser.add_argument('--config', '-c', type=str, default='eval_config.yml',
        help='Path to the evaluation configuration with yaml format.')
    # Read data from args or config.
    parser.add_argument('--data_dir', '-dd', type=str, default=None,
        help='Path to the source dataset, tfrecord | dataset_backbone | folder')

    parser.add_argument('--data_type', '-dt', type=str, default='folder',
        help='Type of the input dataset.')

    args = parser.parse_args()
    main(args)