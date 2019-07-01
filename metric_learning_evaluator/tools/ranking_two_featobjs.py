"""
  Ranking Evaluation for two given feature objects.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from metric_learning_evaluator.data_tools.feature_object import FeatureObject
from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.data_tools.attribute_table import AttributeTable

from metric_learning_evaluator.builder import EvaluatorBuilder
from metric_learning_evaluator.config_parser.parser import ConfigParser

from metric_learning_evaluator.metrics.ranking_metrics import RankingMetrics

class FeatureObjectMerger(object):
    """Merge two objects into single embedding container with attributes.
    """
    def __init__(self, query_path, database_path,
        query_attr='query', database_attr='database', attr_path=None):

        if not query_path:
            pass
        if not database_path:
            pass

        if attr_path is None:
            self._attr_path = '/tmp/query_database_attr.db'
        else:
            self._attr_path = attr_path

        # remove the origin path
        if os.path.exists(self._attr_path):
            os.remove(self._attr_path)

        self._query_featobj_path = query_path
        self._database_featobj_path = database_path
        self._query_attr = query_attr
        self._database_attr = database_attr

        self._create_container()

    @staticmethod
    def _load_featobj(featobj_path):
        """
          Args:
            path: string
          Returns:
            instance_ids
            embeddings
            label_ids
            label_names
            filenames
        """
        feature_importer = FeatureObject()
        feature_importer.load(featobj_path)

        embeddings = feature_importer.embeddings
        filenames = feature_importer.filename_strings
        instance_ids = feature_importer.instance_ids

        if instance_ids is None or instance_ids.size == 0:
            instance_ids = np.arange(embeddings.shape[0])

        label_ids = feature_importer.label_ids
        label_names = feature_importer.label_names
        probabilities = feature_importer.probabilities
        has_label_name = True if label_names is not None else False
        # TODO
        has_prob = True if probabilities is not None else False

        if not has_label_name:
            # turn label_ids into array of string
            label_names = []

        print('load embeddings:{} from {}'.format(embeddings.shape, featobj_path))

        return instance_ids, embeddings, label_ids, label_names, filenames

    def _create_container(self):
        query_inst_ids, query_feats, query_labels, query_labelnames, query_filenames = self._load_featobj(self._query_featobj_path)
        db_inst_ids, db_feats, db_labels, db_labelnames, db_filenames = self._load_featobj(self._database_featobj_path)

        num_query, dim_query = query_feats.shape
        num_database, dim_database = db_feats.shape
        num_total_features = num_query + num_database

        query_attr = [self._query_attr] * num_query
        database_attr = [self._database_attr] * num_database

        global_instance_ids = np.arange(num_total_features)
        embeddings = np.concatenate((query_feats, db_feats), axis=0)
        label_ids = np.concatenate((query_labels, db_labels), axis=0)
        label_names = np.concatenate((query_labelnames, db_labelnames), axis=0)
        filenames = np.concatenate((query_filenames, db_filenames), axis=0)
        attributes = np.concatenate((query_attr, database_attr), axis=0)

        assert dim_query == dim_database, 'Feature dimension of two given featobj should be the same!'
        self._embedding_size = dim_query
        self._num_features = num_total_features

        self._container = EmbeddingContainer(dim_query, 0, num_total_features)
        self._attribute_table = AttributeTable(self._attr_path)

        for inst_id, feat, label, name, attr in zip(
            global_instance_ids, embeddings, label_ids, label_names, attributes):
            inst_id = int(inst_id)
            self._container.add(inst_id, label, feat, label_name=name, attributes=attr)
            self._attribute_table.insert_domain(inst_id, attr)
            if inst_id % 1000 == 0:
                print('{} features are added'.format(inst_id))
        self._attribute_table.commit()

    @property
    def container(self):
        return self._container

    @property
    def embedding_size(self):
        return self._embedding_size

    @property
    def num_total_embeddings(self):
        return self._num_features

    @property
    def query_path(self):
        return self._query_path

    @property
    def database_path(self):
        return self._database_path

    @property
    def attribute_path(self):
        return self._attr_path

def main(args):
    query_data_dir = args.data_dir
    database_dir = args.database
    out_dir = args.out_dir

    if query_data_dir is None or database_dir is None:
        raise ValueError('data_dir or database should be assigned.')

    query_attr = args.query_attribute_name if args.query_attribute_name is not None else 'query'
    database_attr = args.database_attribute_name if args.database_attribute_name is not None else 'database'

    merger = FeatureObjectMerger(query_data_dir, database_dir, query_attr, database_attr)

    attribute_path = merger.attribute_path
    container_size = merger.num_total_embeddings
    embedding_size = merger.embedding_size

    config_dict = {
        'database': {
            'database_type': 'Native',
            'database_config': {'path': attribute_path},
        },
        'index_agent': 'HNSW',
        'container_size': container_size,
        'chosen_evaluations': ['RankingEvaluation'],
        'evaluation_options': {
            'RankingEvaluation': {
                'sampling': {
                    'class_sample_method': 'uniform',
                    'instance_sample_method': 'uniform',
                    'num_of_db_instance_per_class': 1000,
                    'num_of_query_class': 'all_class',
                    'num_of_query_instance_per_class': 1000,
                    'maximum_of_sampled_data': 100000,
                },
                'metric': {
                    'top_k_hit_accuracy': [1, 10],
                    'mAP': True,
                },
                'attribute': {
                    'cross_reference': ['{}->{}'.format(query_attr, database_attr)],
                    'group': ['{}'.format(query_attr),
                              '{}'.format(database_attr),
                              '{}+{}'.format(query_attr, database_attr)],
                },
            },
        }
    }

    evaluator = EvaluatorBuilder(embedding_size, 0, config_dict)
    evaluator.add_container(merger.container)

    results = evaluator.evaluate()
    for metric_name in evaluator.metric_names:
        if metric_name in results:
            print('{}: {}'.format(metric_name, results[metric_name]))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Evaluation for Given Database')

    parser.add_argument('-c', '--config_path', type=str, default='config.yml',
                        help='Path to the configuration yaml file.')
    parser.add_argument('-qd', '--data_dir', type=str, default=None,
                        help='Path to Input DatasetBackbone or raw image folder.')
    parser.add_argument('-db', '--database', type=str, default=None)

    parser.add_argument('-qd_attr', '--query_attribute_name', type=str, default=None)
    parser.add_argument('-db_attr', '--database_attribute_name', type=str, default=None)

    parser.add_argument('-od', '--out_dir', type=str, default=None,
                        help='Path to Output DatasetBackbone.')

    args = parser.parse_args()
    main(args)
