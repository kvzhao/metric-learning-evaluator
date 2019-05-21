"""Inference Tool

  Configurations
    - merge two dict if not given

  Provide several task options
    - extract_feature
    - two_stage
    - cropbox
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import yaml
import numpy as np

from metric_learning_evaluator.utils.switcher import switch

from metric_learning_evaluator.inference.app.two_stage_retrieval import retrieval_application
from metric_learning_evaluator.inference.app.agnostic_detection import detection_application
from metric_learning_evaluator.inference.app.sequential_feature_extraction import extraction_application
from metric_learning_evaluator.application.standard_fields import ApplicationStatusStandardFields as status_fields

import argparse

parser = argparse.ArgumentParser('Command-line Metric Learning Inference Tool')

# must-have argument
parser.add_argument('--task', '-t', type=str, default=None,
        help='Task options: extract | detect | two-stage |')

parser.add_argument('--config', '-c', type=str, default=None,
        help='Path to the inference configuration with yaml format.')

# Read data from args or config.
parser.add_argument('--data_dir', '-dd', type=str, default=None,
        help='Path to the source (query) dataset.')

parser.add_argument('--database', '-db', type=str, default=None,
        help='Path to the source dataset, with type folder')

parser.add_argument('--out_dir', '-od', type=str, default=None,
        help='Path to the output dir for saving report.')

parser.add_argument('--data_type', '-dt', type=str, default='datasetbackbone',
        help='Type of given `data_dir`: datasetbackbone | folder')

APP_SIGNATURE = '[INFERENCE]'

COMMAND_OPERATION_MAPPING = {
}

def main():
    args = parser.parse_args()

    task_job = args.task
    config_path = args.config

    if not config_path:
        # TODO @kv: Generate the default config.
        raise ValueError('inference configuration must be assigned!')

    try:
        with open(config_path, 'r') as fp:
            config_dict = yaml.load(fp)
    except:
        raise IOError('Can not load yaml from {}.'.format(config_path))
        # TODO: create default config instead of error.

    for case in switch(task_job):

        if case('extract'):
            print('Execute feature extraction')
            extraction_application(config_dict, args)
            break

        if case('two-stage'):
            print('Execute two stage inference - (detector + feature_extractor)')
            retrieval_application(config_dict, args)
            break

        if case('detect'):
            print('Execute agnostic detection')
            detection_application(config_dict, args)
            break

        if case():
            print('Inference tool: task {} is not defined.'.format(task_job))