import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.data_tools.result_container import ResultContainer
from metric_learning_evaluator.data_tools.feature_object import FeatureObject

from metric_learning_evaluator.metrics.ranking_metrics import RankingMetrics

from metric_learning_evaluator.index.utils import euclidean_distance
from metric_learning_evaluator.index.utils import indexing_array

from metric_learning_evaluator.core.standard_fields import AttributeStandardFields as attribute_fields


def main(args):

    data_dir = args.data_dir
    database_dir = args.database
    out_dir = args.out_dir
    num_sampled_qeury = args.num_sampled_qeury

    if data_dir is None or database_dir is None:
        raise ValueError('data_dir or database should be assigned.')

    qeury_features = FeatureObject()
    db_features = FeatureObject()

    qeury_features.load(data_dir)
    db_features.load(database_dir)

    query_embeddings = qeury_features.embeddings
    query_label_ids = qeury_features.label_ids
    qeury_label_names = qeury_features.label_names

    if num_sampled_qeury is not None:
        print('Sample {} query from given feature_object'.format(num_sampled_qeury))
        sampled_query_indices = np.random.choice(
            np.arange(len(query_label_ids)), num_sampled_qeury, replace=False)
        query_embeddings = query_embeddings[sampled_query_indices]
        query_label_ids = query_label_ids[sampled_query_indices]

    db_embeddings = db_features.embeddings
    db_label_ids = db_features.label_ids
    db_label_names = db_features.label_names

    result_container = ResultContainer()

    for top_k in [1, 5]:
        print('Compute top {} results'.format(top_k))

        if top_k == 1:
            continue

        ranking_metrics = RankingMetrics(top_k)
        hit_arrays = np.empty((query_embeddings.shape[0], top_k), dtype=np.bool)

        for _idx, (_query_embed, _query_label) in enumerate(zip(query_embeddings, query_label_ids)):

            distances = euclidean_distance(_query_embed, db_embeddings)

            indexed_retrieved_labels = indexing_array(distances, db_label_ids)

            hits = indexed_retrieved_labels[:top_k] == _query_label

            hit_arrays[_idx, ...] = hits

            ranking_metrics.add_inputs(hit_arrays)
            result_container.add(attribute_fields.all_classes, 'top_k_hit_accuracy',
                                     ranking_metrics.topk_hit_accuracy, condition={'k': top_k})
            if _idx % 1000 == 0:
                print('{} features are computed'.format(_idx))

        result_container.add(attribute_fields.all_classes, 'top_k_hit_accuracy',
                                 ranking_metrics.top1_hit_accuracy, condition={'k': 1})
        result_container.add(attribute_fields.all_classes, 'mAP',
                                 ranking_metrics.mean_average_precision)
    print(result_container.results)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Evaluation for Given Database')

    parser.add_argument('-c', '--config_path', type=str, default='config.yml',
                        help='Path to the configuration yaml file.')
    parser.add_argument('-dd', '--data_dir', type=str, default=None,
                        help='Path to Input DatasetBackbone or raw image folder.')
    parser.add_argument('-db', '--database', type=str, default=None,
                        help='Path to Input DatasetBackbone or raw image folder.')
    parser.add_argument('-od', '--out_dir', type=str, default=None,
                        help='Path to Output DatasetBackbone.')
    parser.add_argument('-ns', '--num_sampled_qeury', type=int, default=None,
                        help='Number of sampled query instances per class.')

    args = parser.parse_args()
    main(args)
