
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from metric_learning_evaluator.data_tools.result_container import ResultContainer

import unittest

from pprint import pprint


class TestResultContainer(unittest.TestCase):

    def test_dict_structure(self):

        res_container = ResultContainer()

        res_container.add('all', 'acc', 0.7)
        res_container.add('all', 'top-1-acc', 0.5)
        res_container.add('all', 'top-k-acc', 0.8, condition={'top_k': 5})
        res_container.add('all', 'top-k-acc', 1.0, condition={'top_k': 10})
        res_container.add('all', 'acc', 0.5, condition='IoU@0.5')
        res_container.add('all', 'mAP', 0.7, condition={'IoU': 0.5})
        res_container.add('all', 'mAP', 0.6, condition={'IoU': 0.5, 'recall': 0.1})

        pprint(res_container.results)
        # Should match metric_names
        pprint(res_container.flatten)


if __name__ == '__main__':
    unittest.main()
