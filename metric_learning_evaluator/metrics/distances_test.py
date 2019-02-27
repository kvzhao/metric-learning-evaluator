
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


from metrics.distances import pairwise_euclidean_distance
from metrics.distances import batch_euclidean_distances

import numpy as np

import unittest
from scipy.spatial import distance_matrix

class TestDistanceFunctions(unittest.TestCase):

    def test_pairwise_euclidean_distance(self):
        pass

    def test_batch_euclidean_distance(self):
        vec = np.array([[1., 2., 3.]])

        batch = np.array([
            [1., 2., 3.],
            [3., 2., 1.],
            [1., 2., 2.],
            [.0, .0, .0]])
        dist = distance_matrix(vec, batch)

        np.testing.assert_almost_equal(dist[0], 
            batch_euclidean_distances([1.,2.,3.], batch))
        

if __name__ == '__main__':
    unittest.main()