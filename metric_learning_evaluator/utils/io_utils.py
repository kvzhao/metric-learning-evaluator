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

def check_instance_id(inst_id):
    if isinstance(inst_id, str):
        #for postfix in ['.png', '.jpg', '.jpeg', '.JPG']:
        #    fn = fn.replace(postfix, '')
        inst_id = inst_id.replace('.jpg','')
        inst_id = inst_id.replace('.png','')
    inst_id = int(inst_id)
    return inst_id

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
    instance_ids = feature_importer.instance_ids

    labels = feature_importer.label_ids
    label_names = feature_importer.label_names
    probabilities = feature_importer.probabilities
    has_label_name = True if label_names is not None else False
    # TODO
    has_prob = True if probabilities is not None else False

    # pseudo instance_ids
    pseudo_instance_ids = np.arange(embeddings.shape[0])

    if instance_ids is None or instance_ids.size == 0:
        instance_ids = pseudo_instance_ids

    num_feature, dim_feature = embeddings.shape
    if verbose:
        print('{} features with dim-{} are loaded'.format(num_feature, dim_feature))

    # err handling: label_ids.shape == 0

    container = EmbeddingContainer(embedding_size=dim_feature, 
        prob_size=0, container_size=num_feature)
    if not has_label_name:
        for inst_id, feat, label in zip(instance_ids, embeddings, labels):
            # use filename_string as instance_id, convert to integer
            #for postfix in ['.png', '.jpg', '.jpeg', '.JPG']:
            #    fn = fn.replace(postfix, '')
            #pseudo_instance_id = int(fn)
            inst_id = int(inst_id)
            container.add(inst_id, label, feat)
    else:
        for inst_id, feat, label, name in zip(instance_ids, embeddings, labels, label_names):
            inst_id = int(inst_id)
            container.add(inst_id, label, feat, label_name=name)

    return container