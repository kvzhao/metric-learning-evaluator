
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import unittest

from metrics.scores import top_k_accuracy
from metrics.scores import kfold_accuracy

class TestAccuracyScores(unittest.TestCase):


    def test_top_k_accuracy(self):
        logits = [[.9, .1, .2, .3, .4], # top-1
                  [.9, .8, .1, .2, .1], # top-2
                  [.25, .2, .05, .1, .1], # top-5
                  [.0, .0, .5, .0, .1], # top-3
                  [.2, .2, .2, .21, .2], # top-2, uniform
                 ]
        labels    = [0, 1, 2, 3, 4]
        predicted = [0, 0, 1, 2, 3]
        labels = np.asarray(labels)
        logits = np.asarray(logits)
        
        top_1 = top_k_accuracy(logits, labels, 1)
        top_2 = top_k_accuracy(logits, labels, 2)
        top_3 = top_k_accuracy(logits, labels, 3)
        top_5 = top_k_accuracy(logits, labels, 5)

        self.assertEqual(top_1, 0.2, 'Top 1 accuracy is not correct.')
        self.assertEqual(top_2, 0.6, 'Top 2 accuracy is not correct.')
        self.assertEqual(top_3, 0.8, 'Top 3 accuracy is not correct.')
        self.assertEqual(top_5, 1.0, 'Top 5 accuracy is not correct.')

if __name__ == '__main__':
    unittest.main()