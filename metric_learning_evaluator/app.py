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

from metric_learning_evaluator.evaluator.evaluator_builder import EvaluatorBuilder
from metric_learning_evaluator.data_tools.feature_object import FeatureDataObject

import argparse

parser = argparse.ArgumentParser('Offline Metric Learning Evaluation')

parser.add_argument('--config', '-c', type=str, default=None,
        help='Path to the evaluation configuration with yaml format.')
# Read data from args or config.
parser.add_argument('--data_dir', '-dd', type=str, default=None,
        help='Path to the source dataset, tfrecord | dataset_backbone | folder')
parser.add_argument('--data_type', '-dt', type=str, default='folder',
        help='Type of the input dataset.')
parser.add_argument('--out_dir', '-od', type=str, default=None,
        help='Path to the output dir for saving report.')

parser.add_argument('--embedding_size', '-es', type=int, default=2048,
        help='Dimension of the given embeddings.')
parser.add_argument('--logit_size', '-ls', type=int, default=0,
        help='Size of the logit used in container.')


def main():
    args = parser.parse_args()
    config_path = args.config
    data_type = args.data_type
    data_dir = args.data_dir

    if not data_dir:
        raise ValueError('data_dir must be assigned!')

    if not config_path:
        # TODO @kv: Generate the default config.
        raise ValueError('evaluation configuration must be assigned!')

    try:
        with open(config_path, 'r') as fp:
            config_dict = yaml.load(fp)
    except:
        raise IOError('Can not load yaml from {}.'.format(config_path))
        # TODO: create default config instead of error.

    # open file
    evaluator = EvaluatorBuilder(args.embedding_size,
                                 args.logit_size,
                                 config_dict)

    if data_type == 'folder':
        feature_importer = FeatureDataObject()
        feature_importer.load(data_dir)
        embeddings = feature_importer.embeddings
        filenames = feature_importer.filename_strings
        labels = feature_importer.label_ids

    # Add datum through loop
    for feat, label, fn in zip(embeddings, labels, filenames):
        evaluator.add_image_id_and_embedding(fn, label, feat)
    total_results = evaluator.evaluate()

    pprint (total_results)
    print (evaluator.metric_names)
