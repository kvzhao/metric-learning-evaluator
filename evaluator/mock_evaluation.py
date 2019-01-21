"""
    The Example of the EvaluationObject
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from evaluator.evaluation_base import EmbeddingContainer
from evaluator.evaluation_base import AttributeContainer
from evaluator.evaluation_base import MetricEvaluationBase

class MockEvaluation(MetricEvaluationBase):

    def __init__(self, per_eval_config, embedding_container, attribute_container):
        super(MockEvaluation, self).__init__(per_eval_config, 
                                             embedding_container,
                                             attribute_container)

        """Mock Evaluation
            This mock evaluation functional object is mainly used in testing.

            * Check dimension when created
            * Execute 'mock_metric' functions when computation

        """
        print ('Create {}'.format(self._evaluation_name))
        # this will be called at builder

        # NOTE: Would the builder check this first?
        self.attribute_container = AttributeContainer()

    def compute(self):
        _img_ids = self._embedding_container.image_ids
        _embeddings = self._embedding_container.embeddings
        _groups = self._attribute_container.group