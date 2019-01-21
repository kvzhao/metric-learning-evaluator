"""
    Main function for the offline evaluation.

    TODO @kv:
    This program would handle source data from tfrecord or dataset backbone.

    NOTE: for developing
    Quickly build up this program.
        1. Json wrapper
        2. Dataset reader
        3. Metric functions

"""

import os
import sys
import yaml

from evaluator.evaluator_builder import EvaluatorBuilder

def main(args):

    # open file
    with open(args.config, 'r') as fp:
        eval_config = yaml.load(fp)

    evaluator = EvaluatorBuilder(eval_config)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Metric-Learning-Evaluator')

    parser.add_argument('--config', '-c', type=str, default='eval_config.yml',
        help='Path to the evaluation configuration with yaml format.')
    # Read data from args or config.
    parser.add_argument('--data_dir', '-dd', type=str, default=None,
        help='Path to the source dataset, tfrecord | dataset_backbone.')

    args = parser.parse_args()

    main(args)