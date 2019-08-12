"""
    Mode Seperation using Hierarchical Clustering

    Input:
        Saved folder of EmbeddingContainer format.
    Output:
        A CSV file

"""

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import cv2
import shutil
import operator
import numpy as np
import pandas as pd

from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.analysis.hierarchical_grouping import HierarchicalGrouping


def main(args):
    data_dir = args.data_dir
    out_dir = args.out_dir

    if data_dir is None:
        raise ValueError('data_dir or database should be assigned.')

    container = EmbeddingContainer()
    container.load(data_dir)

    hg = HierarchicalGrouping(container)
    df = hg.auto_label_subgroup(label_id=1)
    print(df)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Evaluation for Given Database')

    parser.add_argument('-dd', '--data_dir', type=str, default=None,
                        help='Path to Input DatasetBackbone or raw image folder.')
    parser.add_argument('-od', '--out_dir', type=str, default=None,
                        help='Path to output folder.')
    args = parser.parse_args()
    main(args)
