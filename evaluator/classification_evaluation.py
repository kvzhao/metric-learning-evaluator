"""
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from evaluator.evaluation_base import EmbeddingContainer
from evaluator.evaluation_base import MetricEvaluationBase

class ClassificationStandardFields(object):
    # top k accuracy
    top_k = 'Top_k'

eval_fields = ClassificationStandardFields

class ClassificationEvaluation(MetricEvaluationBase):

    def __init__(self, per_eval_config, embedding_container, attribute_container):
        super(ClassificationEvaluation, self).__init__(per_eval_config, 
                                             embedding_container,
                                             attribute_container)
        """Classification Evaluation
            The evaluation computes accuracy from given logits and return top_k.
            
            If attribute is provided, also check the eval_config.
        """

        print ('Create {}'.format(self._evaluation_name))
        # Allocate a local container for attribute
        
        # Parse the per_eval_config;

        if eval_fields.top_k in per_eval_config:
            _top_k = per_eval_config[eval_fields.top_k]
            # TODO @kv:make sure the given number if legal
            self.top_k = _top_k
        else:
            print ('WARNING: No `Top_k` is given, use top-5 as default.')
            self.top_k = 5

    def compute(self):
        """Compute Accuracy.

            Get compute probabilities from logits and compare with label
            from embedding_container.
        """

        if not isinstance(self._embedding_container.logits,
                         (np.ndarray, np.generic)):
            raise AttributeError('Logits should be provided when {} is performed'.format(self.evaluation_name))
        
        img_ids = self._embedding_container.image_ids
        logits = self._embedding_container.logits
        labels = self._embedding_container.get_label_by_image_ids(img_ids)

        predict_labels = np.argmax(logits, axis=1)

        print (predict_labels, labels)
