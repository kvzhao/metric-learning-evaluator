"""
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from evaluator.evaluation_base import MetricEvaluationBase
from metrics.accuracy_scores import top_k_accuracy
from core.eval_standard_fields import AttributeStandardFields as attr_fields

class ClassificationStandardFields(object):
    # top k accuracy
    top_k = 'Top_k'

eval_fields = ClassificationStandardFields

class ClassificationEvaluation(MetricEvaluationBase):

    def __init__(self, per_eval_config):
        super(ClassificationEvaluation, self).__init__(per_eval_config)
        """Classification Evaluation
            The evaluation computes accuracy from given logits and return top_k.
            If attributes are provided, calculate top_k accuracy per attribute.
        """

        print ('Create {}'.format(self._evaluation_name))
        
        # Parse the per_eval_config;
        if eval_fields.top_k in per_eval_config:
            _top_k = per_eval_config[eval_fields.top_k]
            # TODO @kv:make sure the given number if legal
            if isinstance(_top_k, list):
                self.top_k = _top_k
            elif isinstance(_top_k, int):
                self.top_k = [_top_k]
            else:
                print ('WARNING: Illegal `Top_k` is given, use top-1 and top-5 as default.')
                self.top_k = [1, 5]
        else:
            print ('WARNING: No `Top_k` is given, use top-1 and top-5 as default.')
            self.top_k = [1, 5]

    def compute(self, embedding_container, attribute_container=None):
        """Compute Accuracy.

            Get compute probabilities from logits and compare with label
            from embedding_container.
        """

        if not isinstance(embedding_container.logits,
                         (np.ndarray, np.generic)):
            raise AttributeError('Logits should be provided when {} is performed'.format(self.evaluation_name))
        
        img_ids = embedding_container.image_ids

        # NOTE: Make sure that length of inputs are equal.

        # Evaluate overall if attribute not given
        if not attribute_container:
            logits = embedding_container.logits
            gt_labels = embedding_container.get_label_by_image_ids(img_ids)

            for k in self.top_k:
                acc_k = top_k_accuracy(logits, gt_labels, k)
                eval_results[eval_fields.top_k][k] = acc_k

        # Evaluate Per Attributes
        else:
            _attributes = self._per_eval_config[attr_fields.attr]
            for _attr in _attributes:
                pass
 
        return eval_results
