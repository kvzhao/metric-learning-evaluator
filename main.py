"""
    Main function for the offline evaluation.
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
    parser.add_argument('--tfrecord')

    args = parser.parse_args()

    main(args)