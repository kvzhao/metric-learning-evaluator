
import os
import sys
import numpy as np
import unittest

from classification_metrics import ClassificationMetrics

def random_boolean_array(N, p=0.5):
    # return a boolean numpy array
    mask = np.empty((N, ))
    for i in range (N):
         mask[i] = np.random.choice(a=[False, True], size=1, p=[p, 1-p])            
    return mask


class TestClassificationMetrics(unittest.TestCase):

    def test_add_inputs_and_accuracy(self):

        metrics = ClassificationMetrics()

        num_of_samples = 10000
        mock_predicts = random_boolean_array(num_of_samples)
        mock_labels = random_boolean_array(num_of_samples)

        metrics.add_inputs(mock_predicts, mock_labels)
        print (metrics.accuracy)
        print (metrics.true_positive_rate)
        print (metrics.false_positive_rate)
        print (metrics.validation_rate)

if __name__ == '__main__':
    unittest.main()
