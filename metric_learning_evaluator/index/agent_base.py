"""Index Agent used in Evaluator
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from abc import ABCMeta
from abc import abstractmethod

import numpy as np


class AgentBase(object):
    """
      Base object of index agent
    """

    __metaclass__ = ABCMeta

    def __init__(self, instance_ids, embeddings):
        """
          Args:
            instance_ids: 1D Numpy array
            embeddings: 2D numpy array with shape = (num_feature, dim_feature)
        """
        self._embeddings = np.squeeze(embeddings)
        self._instance_ids = instance_ids
        assert len(self._embeddings.shape) == 2, 'Embedding must be 2D'
        self._num_embedding, self._dim_embedding = self._embeddings.shape

    @abstractmethod
    def search(self, query_embeddings, top_k):
        """Batch query
          Args:
            query_embeddings: a ndarray with shape (K, d)
                where K is the number of queries; d is the dimension of embedding.
            top_k: an int, top-k results
          Returns:
            A tuple of (batch_distances, batch_indices)
        """
        pass