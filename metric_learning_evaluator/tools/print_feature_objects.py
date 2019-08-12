import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))  # noqa

from metric_learning_evaluator.data_tools.feature_object import FeatureObject
from metric_learning_evaluator.inference.utils.read_json_labelmap import read_json_labelmap


def print_arr_shape(name, shape):
    print('{}: shape={}'.format(name, shape))


def main(args):

    data_dir = args.data_dir
    labelmap_path = args.labelmap

    if labelmap_path is not None:
        labelmap = read_json_labelmap(labelmap_path)
    else:
        # fill the map with loaded label_names andd label_ids
        labelmap = {}

    extracted_features = FeatureObject()
    extracted_features.load(data_dir)

    embeddings = extracted_features.embeddings
    filename_strings = extracted_features.filename_strings
    label_ids = extracted_features.label_ids
    label_names = extracted_features.label_names
    probabilities = extracted_features.probabilities

    print_arr_shape('embeddings', embeddings.shape)
    print_arr_shape('filename_strings', filename_strings.shape)
    print_arr_shape('label_ids', label_ids.shape)
    print_arr_shape('label_names', label_names.shape)
    print_arr_shape('probabilities', probabilities.shape)

    """
    for fn, label_id, label_name in zip(filename_strings, label_ids, label_names):
        print(fn, label_name, label_id)
        print(labelmap[label_name]['label_name'], labelmap[label_name]['unique_id'])
    """


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Export embeddings with inference model.')

    parser.add_argument('-dd', '--data_dir', type=str, default='extracted_embeddings',
                        help='Path to exported features.')
    parser.add_argument('-lm', '--labelmap', type=str, default=None,
                        help='Path to exported features.')

    args = parser.parse_args()
    main(args)