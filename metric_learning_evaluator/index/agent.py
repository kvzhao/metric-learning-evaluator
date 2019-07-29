
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from metric_learning_evaluator.index.np_agent import NumpyAgent
from metric_learning_evaluator.index.hnsw_agent import HNSWAgent
from metric_learning_evaluator.utils.switcher import switch
from metric_learning_evaluator.core.standard_fields import ConfigStandardFields as config_fields

# TODO: Import from core but conflict
REGISTERED_INDEX_AGENT = {
    config_fields.numpy_agent: NumpyAgent,
    config_fields.hnsw_agent: HNSWAgent,
}


class IndexAgent:
    # Wrapper of the IndexAgent

    def __init__(self,
                 agent_type,
                 instance_ids,
                 embeddings,
                 distance_measure='l2',
                 ef_construction=200,
                 num_threads=4,
                 M=32):
        if agent_type not in REGISTERED_INDEX_AGENT:
            raise ValueError('index agent:{} is not registerred'.format(agent_type))
        else:
            self.agent_type = agent_type

        self.agent_engine = REGISTERED_INDEX_AGENT[self.agent_type](
            instance_ids, embeddings, distance_measure, ef_construction, num_threads, M)

    def search(self, query_embeddings, top_k=10):
        return self.agent_engine.search(query_embeddings, top_k)

    def distance_matrix(self, target=None):
        # Notice: This function is time and space consuming.
        if target is None:
            target_embeddings = self.agent_engine._embeddings
        else:
            target_embeddings = target
        indices, distances = self.agent_engine.search(
            target_embeddings, self.agent_engine._num_embedding)
        for i, (idx, dist) in enumerate(zip(indices, distances)):
            distances[i, ...] = distances[i, np.argsort(idx)]
        return distances
