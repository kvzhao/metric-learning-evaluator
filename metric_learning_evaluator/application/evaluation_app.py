"""ml-evaluation

    Main function for the offline evaluation.
    This program load extracted features and analyze results.

    Input:
        Saved EmbeddingContainer (with a folder)
        ├── attribute.db
        ├── attribute_table.csv
        ├── embeddings.npy
        ├── filename_strings.npy
        ├── instance_ids.npy
        ├── label_ids.npy
        └── label_names.npy

    Functions:
        - Evaluate single container
        - Evaluate two containers

    Output:

    Post-Analysis:


    TODO:
        - Auto configuration
        - Logger & Error handling
        - Evaluation & Result container working relation
"""

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import yaml
import numpy as np

from metric_learning_evaluator.builder import EvaluatorBuilder

from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.utils.switcher import switch

# should cooperate add_container
from metric_learning_evaluator.utils.io_utils import check_instance_id
from metric_learning_evaluator.utils.report_writer import ReportWriter

from metric_learning_evaluator.core.standard_fields import ApplicationStatusStandardFields as status_fields
from metric_learning_evaluator.core.standard_fields import EvaluationStandardFields as metric_fields

from metric_learning_evaluator.core.registered import EVALUATION_DISPLAY_NAMES as display_namemap

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

parser.add_argument('--data_type', '-dt', type=str, default='embedding_container',
                    help='Type of the input dataset, Future supports: embedding_container | embedding_db')

parser.add_argument('--out_dir', '-od', type=str, default=None,
                    help='Path to the output dir for saving report.')
parser.add_argument('--embedding_size', '-es', type=int, default=2048,
                    help='Dimension of the given embeddings.')
# score_size, prob_size
parser.add_argument('--prob_size', '-ps', type=int, default=0,
                    help='Size of the output probability size used in container, set 0 to disable')

parser.add_argument('--verbose', '-v', action='store_true')
APP_SIGNATURE = '[EVAL]'


def main():
    args = parser.parse_args()
    config_path = args.config
    data_type = args.data_type
    data_dir = args.data_dir
    out_dir = args.out_dir
    database_dir = args.database

    status = status_fields.not_determined

    if not data_dir:
        raise ValueError('data_dir must be assigned!')

    # argument logic
    if data_dir and database_dir:
        status = status_fields.evaluate_query_database
        raise NotImplementedError('Query to Database function is not implemented yet.')
    elif data_dir and database_dir is None:
        status = status_fields.evaluate_single_container

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
                                 args.prob_size,
                                 config_dict,
                                 mode='offline')

    container = EmbeddingContainer()
    if data_type == 'embedding_container':
        container.load(data_dir)
    elif data_type == 'embedding_db':
        container.load_pkl(data_dir)

    print('evaluator metric names: {}'.format(evaluator.metric_names))

    for case in switch(status):
        print('{} Executes {}'.format(APP_SIGNATURE, status))

        if case(status_fields.evaluate_single_container):
            # Add datum through loop

            evaluator.add_container(container)
            all_measures = evaluator.evaluate()

            print('----- evaluation results -----')
            # remove this
            for metric_name in evaluator.metric_names:
                if metric_name in all_measures:
                    print('{}: {}'.format(metric_name, all_measures[metric_name]))

            if out_dir is not None:
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

            for eval_name, container in evaluator.results.items():
                print(eval_name)
                display_name = display_namemap[eval_name] if eval_name in display_namemap else eval_name
                reporter = ReportWriter(container)
                overall_report = reporter.overall_report
                print(overall_report)
                if out_dir:
                    path = '/'.join([out_dir, 'result_{}'.format(display_name)])
                    container.save(path)

            # end of switch case
            break
