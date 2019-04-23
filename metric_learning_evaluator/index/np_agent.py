import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.index.agent_base import AgentBase

import numpy as np

def euclidean_distance(matrixA, matrixB):
    """
      Args:
        matrixA: 2D numpy array
        matrixB: 2D numpy array
      Return:
        distances: 
    """
    if not isinstance(matrixA, (np.generic, np.ndarray)):
        matrixA = np.asarray(matrixA)
    if not isinstance(matrixB, (np.generic, np.ndarray)):
        matrixB = np.asarray(matrixB)

    distances = np.sum(np.square(np.subtract(matrixA, matrixB)), axis=1)
    return distances

def angular_distance(vectorA, vectorB):
    """
      Args:
        vectorA : 1D numpy array
        vectorB : 1D numpy array
      Return:
        angle (degrees) between two vectors : Float
    """
    def _unit_vector(vector):
        return vector / np.linalg.norm(vector)

    v1_u = _unit_vector(vectorA)
    v2_u = _unit_vector(vectorB)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def indexing_array(distances, target_array):
    """Sort target array according to distances
      Args:
        distances:
        target_array:
      Returns:
        sorted_target
    """
    return target_array[distances.argsort()]

class NumpyAgent(AgentBase):
    def __init__(self,
                 embedding_container: EmbeddingContainer):
        super(NumpyAgent, self).__init__(embedding_container)

        self._build()

    def _build(self):
        self._indices = np.asarray(self._container.instance_ids)

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
        database_size = self._container.counts
        if top_k is None or top_k > database_size:
            top_k = database_size

        batch_distances = np.empty((batch_size, top_k), dtype=np.float32)
        batch_indices = np.empty((batch_size, top_k), dtype=np.float32)

        for batch_idx, query_embed in enumerate(query_embeddings):
            distances = euclidean_distance(query_embed, self._container.embeddings)

            sorted_distances = indexing_array(distances, distances)
            sorted_indices = indexing_array(distances, self._indices)

            batch_distances[batch_idx, ...] = sorted_distances[:top_k]
            batch_indices[batch_idx, ...] = sorted_indices[:top_k]

        return (batch_distances, batch_indices)
