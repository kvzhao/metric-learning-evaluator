import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import unittest
from metric_learning_evaluator.index.hnsw_agent import HNSWAgent
from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer



def fill_embedding_container(num_elements = 10000, dim = 2048):
    # return the embedding_container

    container = EmbeddingContainer(
        embedding_size=dim,
        logit_size=0,
        container_size=num_elements)
    mock_features = np.random.random((num_elements, dim))
    mock_instances = np.arange(num_elements)

    for _idx in range(num_elements):
        inst_id = int(mock_instances[_idx])
        label_id = int(np.random.rand(100)[0])
        container.add(inst_id, label_id, mock_features[_idx])

    return container

class TestHNSWAgent(unittest.TestCase):

    def test(self):

        num_elements = 100
        num_query = 10
        dim = 128

        embedding_container = fill_embedding_container(num_elements, dim)

        agent = HNSWAgent(embedding_container)

        qeury_features = np.random.random((num_query, dim))

        dist, indices = agent.search(qeury_features, 100)
        print(dist.shape)
        print(indices.shape)

if __name__ == '__main__':
    unittest.main()