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

from metric_learning_evaluator.analysis.manifold import Manifold
from metric_learning_evaluator.core.standard_fields import AttributeStandardFields as attribute_fields

from collections import Counter
from pprint import pprint

import pickle


def main(args):
    data_dir = args.data_dir
    out_dir = args.out_dir

    if data_dir is None:
        raise ValueError('data_dir or database should be assigned.')

    feature_object = FeatureObject()
    feature_object.load(data_dir)

    embeddings = feature_object.embeddings
    embeddings = np.squeeze(embeddings)
    filename_strings = feature_object.filename_strings
    label_ids = feature_object.label_ids
    label_names = feature_object.label_names
    instance_ids = np.arange(embeddings.shape[0])

    # Push all embeddings into container
    embedding_container = EmbeddingContainer(
        embedding_size=embeddings.shape[1],
        prob_size=0,
        container_size=embeddings.shape[0])

    for emb, inst_id, label_id in zip(embeddings, instance_ids, label_ids):
        embedding_container.add(inst_id, label_id, emb)

    manifold = Manifold(embedding_container, label_names)
    centers = manifold.class_center()

    c2c_matrix = manifold.center_to_center_relation()
    c2all_relation = manifold.center_to_all_instance_relation()

    for center_label, center_feature in centers.items():
        manifold.distance_trace(center_label, center_feature, 200)

    manifold.locality_analysis()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Evaluation for Given Database')

    parser.add_argument('-dd', '--data_dir', type=str, default=None,
                        help='Path to Input DatasetBackbone or raw image folder.')
    parser.add_argument('-od', '--out_dir', type=str, default=None,
                        help='Path to output folder.')

    args = parser.parse_args()
    main(args)