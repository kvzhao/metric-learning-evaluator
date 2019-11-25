"""ml-evaluation

    Main function for the offline evaluation.
    This program load extracted features and analyze results.

    Input:
        Saved EmbeddingContainer (with a folder)
        ├── attribute_table.csv
        ├── embeddings.npy
        ├── filename_strings.npy
        ├── indexes.csv
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
from metric_learning_evaluator.utils.report_writer import ReportWriter

from metric_learning_evaluator.core.standard_fields import ApplicationStatusStandardFields as status_fields
from metric_learning_evaluator.core.standard_fields import EvaluationStandardFields as metric_fields
from metric_learning_evaluator.core.standard_fields import ConfigStandardFields as config_fields

from metric_learning_evaluator.core.registered import EVALUATION_DISPLAY_NAMES as display_namemap

import argparse

parser = argparse.ArgumentParser('Command-line Metric Learning Evaluation Tool')

# must-have argument
parser.add_argument('--config', '-c', type=str, default=None,
                    help='Path to the evaluation configuration with yaml format.')

parser.add_argument('--data_type', '-dt', type=str, default='embedding_container',
                    help='Type of the input data, supports embedding_container (folder) | embedding_db (.pkl)')

# Read data from args or config.
parser.add_argument('--data_dir', '-dd', type=str, default=None,
                    help='Path to the source (query) data folder.')

parser.add_argument('--anchor_database', '-ad', type=str, default=None,
                    help='Path to the anchor dataset. Default None, use as gallery as given')

parser.add_argument('--out_dir', '-od', type=str, default=None,
                    help='Path to the output folder for saving outcomes and report.')

parser.add_argument('--embedding_size', '-es', type=int, default=128,
                    help='Dimension of the given embeddings.')

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
    anchor_database_dir = args.anchor_database

    status = status_fields.not_determined

    # check input is given
    if not data_dir:
        raise ValueError('data_dir must be assigned!')

    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    # argument logic
    if data_dir and anchor_database_dir:
        status = status_fields.evaluate_query_anchor
    elif data_dir and anchor_database_dir is None:
        status = status_fields.evaluate_single_container

    if not config_path:
        # TODO: @kv: Generate the default config.
        raise ValueError('evaluation configuration must be assigned!')
    try:
        with open(config_path, 'r') as fp:
            config_dict = yaml.load(fp)
    except:
        raise IOError('Can not load yaml from {}.'.format(config_path))
        # TODO: create default config instead of error.

    # Prepare data container
    container = None
    for case in switch(status):
        print('{} Executes {}'.format(APP_SIGNATURE, status))
        if case(status_fields.evaluate_single_container):
            container = EmbeddingContainer(name='single_container')
            if data_type in ['embedding_container', 'embedding_db']:
                container.load(data_dir)
            # end of switch case
            break

        if case(status_fields.evaluate_query_anchor):
            """TODO: Use native method: merge()
              1. Merge two containers
              2. Add `query->anchor` command in cross_reference
              3. Change number of database
            """

            container = EmbeddingContainer(name='query')
            anchor_container = EmbeddingContainer(name='anchor')
            # load query
            if data_type in ['embedding_container', 'embedding_db']:
                container.load(data_dir)
            # load anchor
            if data_type in ['embedding_container', 'embedding_db']:
                anchor_container.load(anchor_database_dir)

            container.merge(anchor_container,
                            merge_key='merge_record',
                            label_id_rearrange=True)
            # clear buffer
            anchor_container.clear()

            # Change config TODO: A little bit hacky, modify in future
            # TODO: It seems not work well
            _opt = config_fields.evaluation_options
            _rank = 'RankingEvaluation'
            _attr = config_fields.attribute
            _cref = config_fields.cross_reference
            _smp = config_fields.sampling
            _cmd = 'merge_record.query -> merge_record.anchor'
            config_dict[_opt][_rank][_attr][_cref] = list(
                filter(None, config_dict[_opt][_rank][_attr][_cref]))
            if _cmd not in config_dict[_opt][_rank][_attr][_cref]:
                config_dict[_opt][_rank][_attr][_cref].append(_cmd)
            config_dict[_opt][_rank][_smp]['num_of_db_instance_per_class'] = 1000
            # end of switch case
            break

    # Build and run evaluation
    evaluator = EvaluatorBuilder(args.embedding_size,
                                 args.prob_size,
                                 config_dict,
                                 mode='offline')
    print(container)
    evaluator.add_container(container)
    evaluator.evaluate()

    # Show Results
    for eval_name, result_container in evaluator.results.items():
        print(eval_name)
        display_name = display_namemap[eval_name] if eval_name in display_namemap else eval_name
        reporter = ReportWriter(result_container)
        overall_report = reporter.overall_report
        print(overall_report)
        if out_dir:
            path = '/'.join([out_dir, 'result_{}'.format(display_name)])
            result_container.save(path)

    if status == status_fields.evaluate_query_anchor and out_dir:
        path = '/'.join([out_dir, 'merged_container'])
        container.save(path)
