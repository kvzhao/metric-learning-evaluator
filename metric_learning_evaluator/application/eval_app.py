"""
    Main function for the offline evaluation.
    This program load extracted features and analyze results.

    Assumption:
      1. Folder in the structure
        folder
        ├── embeddings.npy
        ├── filename_strings.npy
        └── label_ids.npy

    TODO @kv:
      This program would handle source data from folder (with numpy arrays)
      tfrecord or dataset backbone.
    NOTE:
      support data_dir & database objects are `feat_obj` and `raw_images`. (tfrecord in the future?)
        ->  status flow
      offline can be totally different mode without calling evaluate.
      # control attribute container.

    NOTE: for developing
    Quickly build up this program.
        1. Json wrapper
        2. Dataset reader
        3. Metric functions
"""

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import yaml
import numpy as np

from pprint import pprint
from metric_learning_evaluator.builder import EvaluatorBuilder
from metric_learning_evaluator.data_tools.feature_object import FeatureObject
from metric_learning_evaluator.utils.switcher import switch
# should cooperate add_container
from metric_learning_evaluator.utils.io_utils import fetch_embedding_container_from_featobj


import argparse

parser = argparse.ArgumentParser('Command-line Metric Learning Evaluation Tool')

# must-have argument
parser.add_argument('--config', '-c', type=str, default=None,
        help='Path to the evaluation configuration with yaml format.')

# Read data from args or config.
parser.add_argument('--data_dir', '-dd', type=str, default=None,
        help='Path to the source (query) dataset.')
parser.add_argument('--database', '-db', type=str, default=None,
        help='Path to the source dataset, with type folder')
parser.add_argument('--data_type', '-dt', type=str, default='folder',
        help='Type of the input dataset, Future supports: tfrecord | dataset_backbone | folder')
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
    database_dir = args.database

    if not data_dir:
        raise ValueError('data_dir must be assigned!')

    if data_dir and database_dir:
        print('Both query and database are given.')
        raise NotImplementedError('Query to Database function is not implemented yet.')

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
                                 config_dict,
                                 mode='offline')

    if data_type == 'folder':
        feature_importer = FeatureObject()
        feature_importer.load(data_dir)
        embeddings = feature_importer.embeddings
        filenames = feature_importer.filename_strings
        labels = feature_importer.label_ids

    print('evaluator metric names: {}'.format(evaluator.metric_names))
    # Add datum through loop
    for feat, label, fn in zip(embeddings, labels, filenames):
        # TODO @kv: Do not confuse `filename` with `instance_id`.
        fn = fn.replace('.jpg','')
        fn = fn.replace('.png','')
        instance_id = int(fn)
        evaluator.add_instance_id_and_embedding(instance_id, label, feat)
    total_results = evaluator.evaluate()

    for metric_name in evaluator.metric_names:
        print('{}: {}'.format(metric_name, total_results[metric_name]))