"""
    Define data containers for the metric learning evaluator.

    Brief intro:

        EmbeddingContainer: 
            Efficient object which handles the shared (globally) embedding vectors.

        AttributeContainer:
            Data object for maintaining attribute table in each EvaluationObject.

    @bird, dennis, kv
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from abc import ABCMeta
from abc import abstractmethod
import collections
from collections import defaultdict
from metric_learning_evaluator.data_tools.feature_object import FeatureObject


class EmbeddingContainer(object):
    """The Data Container for Embeddings & Logit (instance_id, label_id, embedding vector).

      operations:
        - add: put one datum in the container
        - embeddings: get all embedding vectors exist in the container
        - get_embedding_by_instance_ids: query embeddings by instance_ids
        - get_label_by_instance_ids: query labels by instance_ids
        - clear: clear the internal buffer
    
      NOTE: We CAN NOT confirm the orderness of logits & embedding consistent with instance_ids.
      TODO @kv: implement save & load for data container.
      TODO @kv: Error-handling when current exceeds container_size
      TODO @kv: instance_id can be `int` or `filename`, this is ambiguous
      TODO @kv: logits --> scores (used for classifier)

    """
    def __init__(self, embedding_size, logit_size,
                 container_size=10000, name='embedding_container'):
        """Constructor of the Container.

          Args:
            embedding_size, int:
                Dimension of the embedding vector, e.g. 1024 or 2048.
            logit_size, int:
                Disable this by giving size equals to 0.
            probabilities: an ndarray of probabilities each class, disable this by giving size equals to 0.
                It prefers passing top_k scores.
            container_size, int:
                Number of embedding vector that container can store.
            name, str:
                The name string is used for version control.
        
        """
        self._embedding_size = embedding_size
        self._logit_size = logit_size
        self._container_size = container_size
        # logits, prelogits (embeddeing),
        self._embeddings = np.empty((container_size, embedding_size), dtype=np.float32)
        if logit_size == 0:
            self._logits = None
        else:
            self._logits = np.empty((container_size, logit_size), dtype=np.float32)
        self._label_by_instance_id = {}
        self._index_by_instance_id = {}
        self._instance_id_by_label = defaultdict(list)
        # orderness is maintained in _instance_ids
        self._instance_ids = []
        self._label_ids = []

        self._name = name
        self._current = 0

    def __repr__(self):
        _content = '===== {} =====\n'.format(self._name)
        _content += 'embeddings: {}'.format(self._embeddings.shape)
    
    def add(self, instance_id, label_id, embedding, logit=None):
        """Add instance_id, label_id and embeddings.
        TODO: Add one more argument: logit
          Args:
            instance_id, int:
                Unique instance_id which can not be repeated in the container.
            label_id, int:
                Index of given class corresponds to the instance.
            embedding, numpy array:
                One dimensional embedding vector with size less than self._embedding_size.
            (optional) logit, numpy array:
                One dimensional vector.
        """

        # assertions: embedding size, 
        assert embedding.shape[0] <= self._embedding_size, "Size of embedding vector is greater than the default."
        # TODO @kv: Also check the logit size, and if it exists.

        # NOTE @kv: Do we have a better round-off?
        assert self._current < self._container_size, "The embedding container is out of capacity!"

        if not isinstance(embedding, (np.ndarray, np.generic)):
            raise TypeError ('Legal dtype of embedding is numpy array.')

        self._embeddings[self._current, ...] = embedding

        if not logit is None:
            self._logits[self._current, ...] = logit

        # check type of label_id, instance_id
        try:
            label_id = int(label_id)
            instance_id = int(instance_id)
        except:
            raise TypeError("The label id or instance id has wrong type")

        # NOTE: same instance_id maps to many embedding!?
        self._index_by_instance_id[instance_id] = self._current
        self._label_by_instance_id[instance_id] = label_id
        self._instance_id_by_label[label_id].append(instance_id)
        self._instance_ids.append(instance_id)
        self._label_ids.append(label_id)

        self._current += 1

    def get_embedding_by_instance_ids(self, instance_ids):
        """Fetch batch of embedding vectors by given instance ids."""
        if not (type(instance_ids) is int or type(instance_ids) is list):
            if isinstance(instance_ids, (np.ndarray, np.generic)):
                instance_ids = instance_ids.tolist()
            else:
                raise ValueError('instance_ids should be int or list.')
        if isinstance(instance_ids, int):
            instance_ids = [instance_ids]
        indices = [self._index_by_instance_id[img_id] for img_id in instance_ids]
        return self._embeddings[indices, ...]

    def get_embedding_by_label_ids(self, label_ids):
        """Fetch batch of embedding vectors by given label ids."""
        if not (type(label_ids) is int or type(label_ids) is list):
            raise ValueError('instance_ids should be int or list.')
            if isinstance(label_ids, (np.ndarray, np.generic)):
                label_ids = label_ids.tolist()
            else:
                raise ValueError('instance_ids should be int or list.')
        if isinstance(label_ids, int):
            label_ids = [label_ids]
        
        indices = []
        for label_id in label_ids:
            for inst_id in self.get_instance_ids_by_label(label_id):
                indices.append(self._index_by_instance_id[inst_id])
        return self._embeddings[indices, ...]

    def get_label_by_instance_ids(self, instance_ids):
        """Fetch the labels from given instance_ids."""
        if isinstance(instance_ids, list):
            return [self._label_by_instance_id[img_id] for img_id in instance_ids]
        elif isinstance(instance_ids, int):
            return self._label_by_instance_id[instance_ids]
        elif isinstance(instance_ids, (np.ndarray, np.generic)):
            return [self._label_by_instance_id[img_id] for img_id in instance_ids.tolist()]
        else:
            raise TypeError('instance_ids should be int, list or array.')

    def get_instance_ids_by_label(self, label_id):
        """Fetch the instance_ids from given label_id."""
        if not np.issubdtype(type(label_id), np.integer):
            raise ValueError('Query label id should be integer.')
        return self._instance_id_by_label[label_id]

    def get_instance_ids_by_exclusive_label(self, label_id):
        """Fetch instance_ids except given label_id."""
        if not np.issubdtype(type(label_id), np.integer):
            raise ValueError('Query label id should be integer.')
        exclusive_label_ids = [_id for _id in self._label_ids if _id != label_id]
        return self.get_instance_ids_by_label_ids(exclusive_label_ids)

    def get_instance_ids_by_label_ids(self, label_ids):
        """Fetch the instance_ids from given label_id."""
        if not (type(label_ids) is int or type(label_ids) is list):
            raise ValueError('instance_ids should be int or list.')
        if isinstance(label_ids, int):
            label_ids = [label_ids]
        _instance_ids = []
        for label_id in label_ids:
            _instance_ids.extend(self._instance_id_by_label[label_id])
        return _instance_ids

    @property
    def embeddings(self):
        # get embeddings up to current index
        return self._embeddings[:self._current]

    @property
    def logits(self):
        # get logits up to current index
        return self._logits[:self._current]

    @property
    def instance_ids(self):
        # get all instance_ids in container
        return self._instance_ids
    @property
    def label_ids(self):
        return self._label_ids

    @property
    def instance_id_groups(self):
        return self._instance_id_by_label

    @property
    def index_by_instance_ids(self):
        return self._index_by_instance_id

    @property
    def embedding_size(self):
        return self._embedding_size

    @property
    def counts(self):
        return self._current

    def clear(self):
        # clear dictionaries
        self._index_by_instance_id = {}
        self._label_by_instance_id = {}
        self._instance_ids = []
        
        # reset the current index, rewrite array instead of clear elements.
        self._current = 0
        print ('Clear embedding container.')

    def save(self, path):
        pass

    def load(self, path):
        pass
