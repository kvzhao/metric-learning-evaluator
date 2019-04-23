import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


import numpy as np

from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.data_tools.result_container import ResultContainer
from metric_learning_evaluator.data_tools.feature_object import FeatureDataObject

from metric_learning_evaluator.metrics.ranking_metrics import RankingMetrics

from metric_learning_evaluator.utils.distances import euclidean_distance
from metric_learning_evaluator.utils.distances import indexing_array

from metric_learning_evaluator.tools.manifold import Manifold
from metric_learning_evaluator.query.standard_fields import AttributeStandardFields as attribute_fields

from collections import Counter
from pprint import pprint

import pickle

import matplotlib.pyplot as plt

def main(args):
    data_dir = args.data_dir
    out_dir = args.out_dir

    if data_dir is None:
        raise ValueError('data_dir or database should be assigned.')

    feature_object = FeatureDataObject()
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
        logit_size=0,
        container_size=embeddings.shape[0])

    for emb, inst_id, label_id in zip(embeddings, instance_ids, label_ids):
        embedding_container.add(inst_id, label_id, emb)

    manifold = Manifold(embedding_container, label_names)
    
    if args.label_id is not None:
        intra_class_angles, inter_class_angles = manifold.one_class_pairwise_relation(label_id=args.label_id)
    else:
        intra_class_angles, inter_class_angles = manifold.all_pairwise_relation()

    plt.hist(intra_class_angles, bins=100, alpha=0.5, density=True) 
    plt.hist(inter_class_angles, bins=100, alpha=0.5, density=True)
    if args.label_id:
        plt.title(manifold._labelmap[args.label_id])
    plt.legend(['positive pairs','negative pairs'])
    plt.xlabel('degrees')
    plt.savefig("angles.png")
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Evaluation for Given Database')

    parser.add_argument('-c', '--config_path', type=str, default='config.yml',
                        help='Path to the configuration yaml file.')
    parser.add_argument('-dd', '--data_dir', type=str, default=None,
                        help='Path to Input DatasetBackbone or raw image folder.')
    parser.add_argument('-od', '--out_dir', type=str, default=None,
                        help='Path to output folder.')
    parser.add_argument('-lid', '--label_id', type=int, default=None,
                        help="The label id to be analyzed")

    args = parser.parse_args()
    main(args)