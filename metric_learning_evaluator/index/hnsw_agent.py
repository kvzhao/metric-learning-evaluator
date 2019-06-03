import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from metric_learning_evaluator.index.agent_base import AgentBase
import hnswlib
import numpy as np


class HNSWAgent(AgentBase):
    """Hierachical Navigable Small World search algorithm

      Algorithm parameters can be found:
        https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
    """

    def __init__(self,
                 instance_ids,
                 embeddings,
                 distance_measure='l2',
                 ef_construction=200,
                 num_threads=4,
                 M=32):
        super(HNSWAgent, self).__init__(instance_ids, embeddings)

        self._distance_measure = distance_measure
        self._num_threads = num_threads
        self._ef_construction = ef_construction
        self._M = M

        self._build()
        print('HNSW Index Agent is initialized with {} features'.format(
            self._num_embedding))

    def _build(self):
        """Build search engine and index
        """
        assert len(self._embeddings.shape) == 2, 'Embedding must be 2D'
        self.engine = hnswlib.Index(space=self._distance_measure,
                                    dim=self._dim_embedding)
        self.engine.init_index(max_elements=self._num_embedding,
                               ef_construction=self._ef_construction, M=self._M)

        self.engine.add_items(self._embeddings, self._instance_ids)

        self.engine.set_ef(50)
        self.engine.set_num_threads(self._num_threads)

    def search(self, query_embeddings, top_k=10):
        """Batch query
          Args:
            query_embeddings: a ndarray with shape (K, d)
                where K is the number of queries; d is the dimension of embedding.
            top_k: an int, top-k results
          Returns:
            A tuple of (batch_indices, batch_distances)
        """
        # NOTE: preprocessing and check
        return self.engine.knn_query(query_embeddings, k=top_k)