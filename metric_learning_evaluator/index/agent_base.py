"""Index Agent used in Evaluator
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from abc import ABCMeta
from abc import abstractmethod

from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer


class AgentBase(object):
    """
      Base object of index agent
    """

    __metaclass__ = ABCMeta

    def __init__(self, embedding_container: EmbeddingContainer):
        """
          Args:
            embedding_container:
        """
        self._container = embedding_container

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