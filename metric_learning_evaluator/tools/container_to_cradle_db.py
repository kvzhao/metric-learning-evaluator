"""Script:
  * Cradle EmbeddingDB & EmbeddingContainer Converter
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from collections import defaultdict
from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer

from cradle.data_container.embedding import Embedding
from cradle.data_container.embedding_db import EmbeddingDB

from pprint import pprint


def container_to_cradle(path_container, path_cradle):
    container = EmbeddingContainer()
    container.load(path_container)
    print('Embedding container loaded.')
    print(container)
    embedding_db = EmbeddingDB()
    embedding_db.set(embeddings=container.embeddings,
                     meta_dict=container.meta_dict)

    if not os.path.exists(path_cradle):
        os.makedirs(path_cradle)
    pkl_path = os.path.join(path_cradle, 'embedding_db.pkl')
    md5_path = os.path.join(path_cradle, 'embedding_db_md5.txt')

    embedding_db.dump(pkl_path, md5_path)
    print('Dump embedding_db to {}'.format(path_cradle))


def main(args):
    if args.container is None or args.cradle is None:
        raise('Input paths should not be None.')
    path_embedding_container = args.container
    path_cradle_embedding_db = args.cradle
    container_to_cradle(path_embedding_container,
                        path_cradle_embedding_db)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('EmbeddingContainer to EmbeddingDB Converter')
    parser.add_argument('--container', type=str, default=None, help='Path to EmbeddingContainer (.npy)')
    parser.add_argument('--cradle', type=str, default=None, help='Path to EmbeddingDB (.pkl)')
    args = parser.parse_args()
    main(args)
