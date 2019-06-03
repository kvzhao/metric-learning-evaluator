
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import unittest
import time
from metric_learning_evaluator.index.agent import IndexAgent

class TestAgent(unittest.TestCase):

    def test(self):

        num_elements = 8000
        num_query = 1
        dim = 128

        #embedding_container = fill_embedding_container(num_elements, dim)
        mock_features = np.random.random((num_elements, dim))
        mock_instances = np.arange(num_elements)
        qeury_features = np.random.random((num_query, dim))

        np_agent = IndexAgent('Numpy', mock_instances, mock_features)
        start_time = time.time()
        dist, indices = np_agent.search(qeury_features, 1000)
        end_time = time.time()
        print('Numpy Agent takes {} ms to search'.format((end_time-start_time) * 1000.0))

        hnsw_agent = IndexAgent('HNSW', mock_instances, mock_features)
        start_time = time.time()
        dist, indices = hnsw_agent.search(qeury_features, 1000)
        end_time = time.time()
        print('HNSW Agent takes {} ms to search'.format((end_time-start_time) * 1000.0))

if __name__ == '__main__':
    unittest.main()