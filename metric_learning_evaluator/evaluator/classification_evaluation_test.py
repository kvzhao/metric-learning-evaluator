
import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from metric_learning_evaluator.evaluator.data_container import EmbeddingContainer
from metric_learning_evaluator.evaluator.evaluation_base import MetricEvaluationBase
from metric_learning_evaluator.evaluator.classification_evaluation import ClassificationEvaluation

import unittest

class TestClassificationEvaluation(unittest.TestCase):
    """
      Format of per_eval_config:

    """

    def test_compute_without_attributes(self):
        per_eval_config = []
        num_of_samples = 3
        embedding_size = 10
        logit_size = 3

        embeddings = np.random.rand(num_of_samples, embedding_size)

        logits = [[0.1, 0.5, 0.4],
                  [0.8, 0.1, 0.1],
                  [0.6, 0.3, 0.2]]

        labels = [2, 1, 3]

        embed_container = EmbeddingContainer(embedding_size, logit_size, container_size=10)

        # Iteratively adding datum
        for img_id, (emd, logit, label) in enumerate(zip(embeddings, logits, labels)):
            label_id = label
            embed_container.add(img_id, label_id, emd, logit)


        cls_eval = ClassificationEvaluation(per_eval_config)
        cls_eval.compute(embed_container, None)

if __name__ == '__main__':
    unittest.main()

