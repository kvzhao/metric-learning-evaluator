"""
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from evaluator.evaluation_base import EmbeddingContainer
from evaluator.evaluation_base import MetricEvaluationBase

class ClassificationEvaluation(MetricEvaluationBase):

    def __init__(self, per_eval_config, embedding_container, attribute_container):
        super(ClassificationEvaluation, self).__init__(per_eval_config, 
                                             embedding_container,
                                             attribute_container)

        print ('Create ClassificationEvaluation')
        # Allocate a local container for attribute

    def compute(self):
        pass
        # call metric functions