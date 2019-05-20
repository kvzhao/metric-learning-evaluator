"""IO Utilities used in App
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from metric_learning_evaluator.data_tools.feature_object import FeatureObject
from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer


# TODO @kv : rename
def create_embedding_container_from_featobj(folder_path, verbose=True):
    """Directly load feature object into embedding container.
      Args:
        folder_path: string, path to the folder of FeatureObject
        verbose: Boolean, show the size of feature object if set True
      Return:
        container: EmbeddingContainer
    """
    feature_importer = FeatureObject()
    feature_importer.load(folder_path)
    embeddings = feature_importer.embeddings
    filenames = feature_importer.filename_strings

    labels = feature_importer.label_ids

    # pseudo instance_ids
    instance_ids = np.arange(embeddings.shape[0])

    num_feature, dim_feature = embeddings.shape
    if verbose:
        print('{} features with dim-{} are loaded'.format(num_feature, dim_feature))

    # err handling: label_ids.shape == 0

    container = EmbeddingContainer(embedding_size=dim_feature, 
        logit_size=0, container_size=num_feature)
    for feat, label, fn in zip(embeddings, labels, filenames):
        # use filename_string as instance_id, convert to integer
        for postfix in ['.png', '.jpg', '.jpeg', '.JPG']:
            fn = fn.replace(postfix, '')
        pseudo_instance_id = int(fn)
        container.add(pseudo_instance_id, label, feat)
    return container