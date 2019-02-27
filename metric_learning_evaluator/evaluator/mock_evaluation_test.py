import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


from evaluator.data_container import EmbeddingContainer
from evaluator.data_container import AttributeContainer
from evaluator.mock_evaluation import MockEvaluation

import unittest

class TestMockEvaluation(unittest.TestCase):
    # Define the inputs

    def test_init(self):
        per_eval_config = ["Color", "Shape", "Bottle", "Color", "Shape"]
        desired_attributes = ["Shape", "Bottle", "Color"]

        embed_cont = EmbeddingContainer(18, 8)

        eval_mock = MockEvaluation(per_eval_config)

        self.assertCountEqual(eval_mock._per_eval_config,
                              desired_attributes,
                              'Number of attribute items are not equal.')


if __name__ == '__main__':
    unittest.main()