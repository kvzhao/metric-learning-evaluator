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

    def __init__(self, embedding_container):
        super(MockEvaluation, self).__init__(embedding_container)

        print ('Create MockEvaluation')
        # this will be called at builder
        # NOTE: Would the builder check this first?
        #self.attribute_container = AttributeContainer()

    def compute(self):
        _groups = self.attribute_container.group