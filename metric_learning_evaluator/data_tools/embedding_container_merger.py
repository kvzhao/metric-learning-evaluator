"""
  Embedding Container Merger

  Usage:
    merger = EmbeddingContainerMerger()

    merger.merge(
        [container_1, container_2, container_3]
    )

    merged = merger.emerged_container
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.core.standard_fields import EmbeddingContainerStandardFields as container_fields


class EmbeddingContainerMerger(object):
    """Merger
      operations:
        - fetch internals indices
        - fetch attributes (from attribute table)
        - fetch embeddings (stack numpy array)
        - create a empty container
        - push all data in new container
    """

    def __init__(self, merge_key='type'):
        """
          Args:
            merge_key: string of the merge key
        """
        self._merged_container = None
        self._merge_key = merge_key

    def merge(self, containers):
        """
          Args:
            containers: list of EmbeddingContainer
          Return:
            merged: EmbeddingContainer
          NOTE:
            Init merged only
        """
        # check dimension check type
        # Get embedding size and make sure they are equal
        dim_list = [cont.embedding_size for cont in containers]

        if len(set(dim_list)) != 1:
            raise ValueError('Error, embedding dimension not equal but get {}'.format(dim_list))

        total_num = sum(cont.counts for cont in containers)
        emb_dim = dim_list[0]

        self._merged_container = EmbeddingContainer(emb_dim, 0, total_num)

        # TODO: Make sure all column items are same
        label_ids = []
        label_names = []
        filename_strings = []
        attributes = []
        container_names = []
        global_index = 0

        # push everything onto buffer, then add iteratively
        embeddings = np.vstack([cont.embeddings for cont in containers])

        for container in containers:
            # TODO: Raise if None
            internals = container._fetch_internals()
            _attributes = container._fetch_attributes()

            if not _attributes:
                _attributes = [None] * container.counts

            _label_ids = internals.get(container_fields.label_ids, None)
            _label_names = internals.get(container_fields.label_names, None)
            _filename_strings = internals.get(container_fields.filename_strings, None)

            if _label_ids is None:
                print('WARNING!')

            # extend
            label_ids.extend(_label_ids)
            label_names.extend(_label_names)
            filename_strings.extend(_filename_strings)
            attributes.extend(_attributes)
            container_names.extend([container.name] * container.counts)

        # TODO: @kv Check dimensions are consistent
        for label_id, label_name, filename, attribute, name in zip(label_ids,
                                                                   label_names,
                                                                   filename_strings,
                                                                   attributes,
                                                                   container_names):
            feat = embeddings[global_index]
            if attribute is None:
                attribute = {}
            attribute[self._merge_key] = name
            self._merged_container.add(global_index,
                                       label_id=label_id,
                                       label_name=label_name,
                                       embedding=feat,
                                       attribute=attribute,
                                       filename=filename)
            global_index += 1

        return self.merged_container

    @property
    def merged_container(self):
        if self._merged_container is None:
            return EmbeddingContainer()
        return self._merged_container

    def clear(self):
        self._merged_container = None
