import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

from metric_learning_evaluator.index.agent_base import AgentBase

from metric_learning_evaluator.index.utils import euclidean_distance
from metric_learning_evaluator.index.utils import indexing_array
import numpy as np


class NumpyAgent(AgentBase):
    def __init__(self, instance_ids, embeddings):
        super(NumpyAgent, self).__init__(instance_ids, embeddings)

        self._build()
        print('Numpy Index Agent is initialized')

    def _build(self):
        self._indices = np.asarray(self._instance_ids)

    def search(self, query_embeddings, top_k=None):
        """Batch query
          Args:
            query_embeddings: a ndarray with shape (K, d)
                where K is the number of queries; d is the dimension of embedding.
            top_k: an int, top-k results
          Returns:
            A tuple of (batch_distances, batch_indices)
        """

        batch_size = query_embeddings.shape[0]
        database_size, embedding_size = self._embeddings.shape
        if top_k is None or top_k > database_size:
            top_k = database_size

        batch_distances = np.empty((batch_size, top_k), dtype=np.float32)
        batch_indices = np.empty((batch_size, top_k), dtype=np.float32)

        for batch_idx, query_embed in enumerate(query_embeddings):
            distances = euclidean_distance(query_embed, self._embeddings)

            sorted_distances = indexing_array(distances, distances)
            sorted_indices = indexing_array(distances, self._indices)

            batch_distances[batch_idx, ...] = sorted_distances[:top_k]
            batch_indices[batch_idx, ...] = sorted_indices[:top_k]

        return (batch_distances, batch_indices)
