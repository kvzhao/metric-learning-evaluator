import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from metrics.ranking_metrics import RankingMetrics


class TestRankMetrics(unittest.TestCase):
    """
    Test ranking metrics
    """
    top_k = 5
    valid_inputs = np.array([
        [1, 0, 0, 1, 0, 1, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=bool)
    invalid_shape_inputs = valid_inputs[:, np.newaxis]
    invalid_type_list = [np.int64, np.int32, int]
    invalid_empty_shape_list = [(0, 5), (10, 0), (0, 0)]
    top1_hit_accuracy = 0.4
    topk_hit_accuracy = 0.6
    average_precisions = np.array([
        [0.6212],
        [0.525],
        [0.975],
        [0.3544],
        [0.]],
        dtype=np.float32)
    mean_average_precision = 0.4951
    error_decimal = 4
    error_epsilon = 1e-4

    def test_initialize(self):
        rank_metric_object = RankingMetrics(self.top_k)
        self.assertTrue(
            rank_metric_object.is_empty,
            msg='class is not empty when initialized')
        self.assertEqual(
            rank_metric_object.top_k,
            self.top_k,
            msg='class top_k "{cls}" and argument top_k: "{arg}" is not the same'.format(
                cls=rank_metric_object.top_k,
                arg=self.top_k))

    def test_add_invalid_shape_inputs(self):
        rank_metric_object = RankingMetrics(self.top_k)
        with self.assertRaises(
                ValueError,
                msg='accept invalid input of shape: {}'.format(
                    self.invalid_shape_inputs.shape)):
            self._add_inputs_to_object(rank_metric_object, self.invalid_shape_inputs)

    # TODO @jeff: verify type assertion for np.bool and other types
    # @unittest.expectedFailure
    def test_add_invalid_type_inputs(self):
        rank_metric_object = RankingMetrics(self.top_k)
        for _type in self.invalid_type_list:
            with self.assertRaises(
                    TypeError,
                    msg='accept invalid input type of {}'.format(_type)):
                invalid_type_inputs = self.valid_inputs.astype(_type)
                self._add_inputs_to_object(rank_metric_object, invalid_type_inputs)

    def test_add_empty_inputs(self):
        rank_metric_object = RankingMetrics(self.top_k)
        for _shape in self.invalid_empty_shape_list:
            with self.assertRaises(
                    ValueError,
                    msg='accept empty input of shape {}'.format(_shape)):
                empty_inputs = np.zeros(_shape, dtype=bool)
                self._add_inputs_to_object(rank_metric_object, empty_inputs)

    def test_add_valid_inputs_and_clear(self):
        rank_metric_object = RankingMetrics(self.top_k)
        self._add_inputs_to_object(rank_metric_object, self.valid_inputs)
        self.assertFalse(
            rank_metric_object.is_empty,
            msg='get empty metric object after add inputs to it')
        rank_metric_object.clear()
        self.assertTrue(
            rank_metric_object.is_empty,
            msg='metric object is not empty after clear')

    def test_compute_metrics(self):
        rank_metric_object = RankingMetrics(self.top_k)
        self._add_inputs_to_object(rank_metric_object, self.valid_inputs)
        self.assertAlmostEqual(
            rank_metric_object.top1_hit_accuracy,
            self.top1_hit_accuracy,
            places=self.error_decimal,
            msg='wrong top1 hit accuracy: {wrong} (rather than {correct})'.format(
                wrong=rank_metric_object.top1_hit_accuracy,
                correct=self.top1_hit_accuracy))

        self.assertEqual(
            rank_metric_object.topk_hit_accuracy,
            self.topk_hit_accuracy,
            msg='wrong topk hit accuracy: {wrong} (rather than {correct})'.format(
                wrong=rank_metric_object.topk_hit_accuracy,
                correct=self.topk_hit_accuracy))

        np.testing.assert_array_almost_equal(
            rank_metric_object.average_precisions,
            self.average_precisions,
            decimal=self.error_decimal,
            err_msg='wrong topk hit accuracy: \n{wrong} \nrather than: \n{correct})'
                .format(
                    wrong=rank_metric_object.average_precisions,
                    correct=self.average_precisions))

        self.assertAlmostEqual(
            rank_metric_object.mean_average_precision,
            self.mean_average_precision,
            places=self.error_decimal,
            msg='wrong mean average precision: {wrong} (rather than {correct})'.format(
                wrong=rank_metric_object.mean_average_precision,
                correct=self.mean_average_precision))

    def _add_inputs_to_object(self, metric_object, inputs):
        metric_object.clear()
        metric_object.add_inputs(inputs)


if __name__ == '__main__':
    unittest.main()
