
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import unittest
from core.eval_standard_fields import EvalConfigStandardFields as fields
from evaluator.evaluator_builder import EvaluatorBuilder

class TestEvaluatorBuilder(unittest.TestCase):

    def test_init_datasetbackbone(self):
        """
            Evaluator should be allocates with given
            evaluation list.
        """
        sample_eval_config = {
            fields.database: fields.datasetbackbone,
            #TODO: Modifiy this
            fields.evaluation: {
                fields.mock: {'Attr': ['Color']},
                fields.classification: {},
                fields.ranking: {'Attr': ['Color', 'Shape']},
            },
            fields.container_size: 100000,
            fields.embedding_size: 2048,
            fields.logit_size: 1400,
        }
        list_of_evaluations = [
            'MockEvaluation', 'ClassificationEvaluation', 'RankingEvaluation'
        ]
        eval_object = EvaluatorBuilder(sample_eval_config)

        # Test evaluations are created
        self.assertEqual(eval_object.evaluation_names,
                         list_of_evaluations,
                         'Allocated evaluations are not matched.')

        # Test numpy shape are correct

    def test_init_datasetbackbone_without_logit(self):
        sample_eval_config = {
            fields.database: fields.datasetbackbone,
            fields.evaluation: fields.mock,
            fields.container_size: 100000,
            fields.embedding_size: 2048,
            fields.logit_size: 0,
        }

    def test_add_id_and_images(self):
        """
        """
        sample_eval_config = {
            fields.database: fields.datasetbackbone,
            fields.evaluation: {
                fields.mock: {'Attr': ['Color']},
                fields.classification: {},
                fields.ranking: {'Attr': ['Shape', 'Color', 'Hardness']},
            },
            fields.container_size: 100000,
            fields.embedding_size: 2048,
            fields.logit_size: 0,
        }
        list_of_evaluations = [
            fields.mock,
        ]
        eval_object = EvaluatorBuilder(sample_eval_config)

        # Check the logit is not exist.

    def test_evaluate_classification(self):
        """Test each compute in evaluation.
        """
        num_classes = 6
        num_samples = 5

        sample_eval_config = {
            fields.database: fields.datasetbackbone,
            fields.evaluation: {
                fields.mock: {'Attr': ['Color']},
                fields.classification: {'Top_k': [1, 3, 5]},
                fields.ranking: {'Attr': ['Shape', 'Color', 'Hardness']},
            },
            fields.container_size: 100,
            fields.embedding_size: 10,
            fields.logit_size: num_classes,
        }

        mock_embeddings = np.random.rand(num_samples, 10)
        mock_lables = [0, 1, 2, 3, 4, 3]
        mock_logits = np.asarray(
            [.9, .8, .7, .6, .5, .4],
            [.4, .9, .8, .7, .6, .5],
            [.5, .4, .9, .8, .7, .6],
            [.6, .5, .4, .9, .8, .7],
            [.7, .6, .5, .4, .9, .8],
        )


if __name__ == '__main__':
    unittest.main()