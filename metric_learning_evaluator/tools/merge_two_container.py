"""
  Merger
    - two containers
    - two feature objects
    - one container one feature objects
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.data_tools.embedding_container_merger import EmbeddingContainerMerger


def main(args):
    data_dir_1 = args.data_dir_1
    data_dir_2 = args.data_dir_2
    keyword = args.keyword
    name_data_1 = args.name_1
    name_data_2 = args.name_2
    output_dir = args.output_dir

    qd_container = EmbeddingContainer(name=name_data_1)
    qd_container.load(data_dir_1)

    db_container = EmbeddingContainer(name=name_data_2)
    db_container.load(data_dir_2)

    merger = EmbeddingContainerMerger(keyword)
    merged = merger.merge([qd_container, db_container])

    print(merged)
    print(merged.DataFrame)

    merged.save(output_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Merge two embedding containers')

    parser.add_argument('-d1', '--data_dir_1', type=str,
                        default=None,
                        help='Path to the embedding container folder')

    parser.add_argument('-d2', '--data_dir_2', type=str,
                        default=None,
                        help='Path to the embedding container folder')

    parser.add_argument('-k', '--keyword', type=str,
                        default='type',
                        help='Key of merged attribute')

    parser.add_argument('-n1', '--name_1', type=str,
                        default='query',
                        help='Name of given embedding container')

    parser.add_argument('-n2', '--name_2', type=str,
                        default='anchor',
                        help='Name of given embedding container')

    parser.add_argument('-od', '--output_dir', type=str,
                        default='merged_container',
                        help='Path to the merged embedding container folder')

    args = parser.parse_args()

    main(args)
