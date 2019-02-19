"""
    FacenetEvaluation is the implementation referred to the repo:
        https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw
"""
from __future__ import division

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import math
import random
import itertools
import numpy as np
from core.eval_standard_fields import MetricStandardFields as metric_fields
from core.eval_standard_fields import AttributeStandardFields as attribute_fields

from evaluator.data_container import EmbeddingContainer
from evaluator.data_container import AttributeContainer
from evaluator.data_container import ResultContainer
from evaluator.evaluation_base import MetricEvaluationBase

from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score

# from metrics.scores import calculate_positive_by_distance
from metrics.distances import euclidean_distance
from metrics.distances import euclidean_distance_filter

from metrics.classification_metrics import ClassificationMetrics

from collections import defaultdict
from collections import namedtuple
from collections import Counter
# For verbose
from pprint import pprint

class FacenetEvaluationStandardFields(object):
    # Define fields used in evaluation and executable metrics
    pairA = 'pairA'
    pairB = 'pairB'
    is_same = 'is_same'

    uniform_class = 'uniform_class'
    random_sample = 'random_sample'


facenet_fields = FacenetEvaluationStandardFields

class FacenetEvaluation(MetricEvaluationBase):

    def __init__(self, config):
        """
          FaceNet evaluates accuracy with pair images.

          metric functions:
            - Top_k accuracy of same pairs
            - AUC
            - EER

          distance measures:

          Results
          -------------------------------------------------
            Accuracy: 0.99650+-0.00252
            Validation Rate: 0.98367+-0.00948 @ FAR=0.00100
            Area Under Curve (AUC): 1.000
            Equal Error Rate (EER): 0.004
          -------------------------------------------------
        """
        super(FacenetEvaluation, self).__init__(config)
        
        print ('Create {}'.format(self._evaluation_name))

        # Preprocess Configurations and check legal
        self._available_metrics = [
            metric_fields.accuracy, 
            metric_fields.pair,
            metric_fields.sample_method,
        ]

        self._must_have_metrics = [
            metric_fields.distance_threshold,
            metric_fields.sample_method,
            metric_fields.sample_ratio
        ]

        # TODO: How to set default value?

        self._default_values = {
            metric_fields.distance_threshold: [0.5, 1.0, 1.5],
            metric_fields.sample_method: facenet_fields.uniform_class,
            metric_fields.sample_ratio: 0.2,
        }

        # Set default values for must-have metrics
        for _metric in self._must_have_metrics:
            if not _metric in self._eval_metrics:
                if _metric in self._default_values:
                    self._eval_metrics[_metric] = self._default_values[_metric]
                else:
                    print ("WARNING: {} should be assigned".format(_metric))

        ## attributes
        if len(self._eval_attributes) == 0:
            self._has_attribute = False
        elif len(self._eval_attributes) == 1:
            if attribute_fields.all_classes in self._eval_attributes:
                self._has_attribute = False
            elif attribute_fields.all_attributes in self._eval_attributes:
                self._has_attribute = True
        else:
            self._has_attribute = True

        self.show_configs()



    def compute(self, embedding_container, attribute_container=None):
        """Procedure:
            - prepare the pair list for eval set
            - compute distance
            - calculate accuracy

          NOTE: Attributes are NOT supported yet.
        """

        # numpy array
        img_ids = embedding_container.image_ids

        # configs
        sample_ratio = self._eval_metrics[metric_fields.sample_ratio]
        sample_method = self._eval_metrics[metric_fields.sample_method]
        distance_threshold = self._eval_metrics[metric_fields.distance_threshold]

        assert len(img_ids) == embedding_container.embeddings.shape[0]

        result_container = ResultContainer(self._eval_metrics, self._eval_attributes)

        if not self._has_attribute:
            # attribute == all_classes
            attribute = attribute_fields.all_classes
            # @kv: load pair list from file or generate automatically
            pairs = self._generate_pairs(embedding_container, 
                                         sample_ratio=sample_ratio,
                                         sample_method=sample_method)

            # fetch image ids and compuate distances at once.
            pair_A_embeddings = embedding_container.get_embedding_by_image_ids(
                pairs[facenet_fields.pairA])
            pair_B_embeddings = embedding_container.get_embedding_by_image_ids(
                pairs[facenet_fields.pairB])
            groundtruth_is_same = np.asarray(pairs[facenet_fields.is_same])

            # Distance filtering
            distance_thresholds = self._eval_metrics[metric_fields.distance_threshold]
            predicted_is_same = euclidean_distance_filter(pair_A_embeddings,
                                                          pair_B_embeddings,
                                                          distance_thresholds)

            for threshold in distance_thresholds:
                classification_metrics = ClassificationMetrics()
                classification_metrics.add_inputs(predicted_is_same[threshold], groundtruth_is_same)
                result_container.add(attribute,
                    metric_fields.accuracy, threshold, classification_metrics.accuracy)
                result_container.add(attribute,
                    metric_fields.validation_rate, threshold, classification_metrics.validation_rate)
                result_container.add(attribute, 
                    metric_fields.false_accept_rate, threshold, classification_metrics.false_accept_rate)
                result_container.add(attribute,
                    metric_fields.true_positive_rate, threshold, classification_metrics.true_positive_rate)
                result_container.add(attribute,
                    metric_fields.false_positive_rate, threshold, classification_metrics.false_positive_rate)
        
        else:
            # Has attributes
            pass

        # Fetch results
        print ("{} compute done.".format(self.evaluation_name))

        return result_container.results


    def _generate_pairs(self, embedding_container, sample_ratio=0.2, sample_method='random'):
        """Image Pair & Sampling Strategy

          Args:
            pairs: dict of lists
                Pair with three keys: fields.pairA, fields.pairB, fields.is_same
                e.g.
                    A, B
                    img_1, img_2
                    img_1, img_3
                    img_1, img_4
                    ...
                    img_3, img_4,
                    img_4, img_5
                pairs[is_same]: list of boolean 
                    It denotes corresponding pair is same class or not
            sample_method:
              - random:
              - uniform_class:
        
          Return:
            pairs, dict of list:

        """
        pairs = defaultdict(list)
        image_ids = embedding_container.image_ids
        label_ids = embedding_container.get_label_by_image_ids(image_ids)
        label_counter = Counter(label_ids)
        num_image_ids = len(image_ids)
        num_label_ids = len(label_ids)
        num_class = len(label_counter)

        assert num_label_ids == num_image_ids

        # Randomly sample several data
        image_ids = np.asarray(image_ids)
        label_ids = np.asarray(label_ids)
        num_pair_samples = int(num_image_ids * sample_ratio)

        if sample_method is None or sample_method == facenet_fields.random_sample:
            sampled_image_ids = np.random.choice(image_ids, num_pair_samples)
            for img_a, img_b in list(itertools.combinations(sampled_image_ids, 2)):            
                label_img_a = embedding_container.get_label_by_image_ids(img_a)
                label_img_b = embedding_container.get_label_by_image_ids(img_b)
                is_same = label_img_a == label_img_b
                pairs[facenet_fields.pairA].append(img_a)
                pairs[facenet_fields.pairB].append(img_b)
                pairs[facenet_fields.is_same].append(is_same)  

        elif sample_method == facenet_fields.uniform_class:
            num_samples_per_class = math.floor(num_pair_samples / num_class)
            image_id_groups = embedding_container.image_id_groups

            # Go through all classes
            for class_id, images_in_same_group in image_id_groups.items():
                anchor_image_id = np.random.choice(images_in_same_group, 1, replace=False)[0]
                image_ids_same_group = np.random.choice(images_in_same_group, num_samples_per_class)
                image_ids_diff_group = np.random.choice(image_ids, num_samples_per_class)

                # Image with same label
                for same_img_id in image_ids_same_group:
                    tar_class_id = embedding_container.get_label_by_image_ids(same_img_id)
                    is_same = tar_class_id == class_id
                    pairs[facenet_fields.pairA].append(anchor_image_id)
                    pairs[facenet_fields.pairB].append(same_img_id)
                    pairs[facenet_fields.is_same].append(is_same)  
                    
                # Image with different label
                for diff_img_id in image_ids_diff_group:
                    tar_class_id = embedding_container.get_label_by_image_ids(diff_img_id)
                    is_same = tar_class_id == class_id
                    pairs[facenet_fields.pairA].append(anchor_image_id)
                    pairs[facenet_fields.pairB].append(diff_img_id)
                    pairs[facenet_fields.is_same].append(is_same)  
                
        return pairs


    def _compute_roc(self):
        """ROC Curve
        """
        pass

    def _compute_err_rate(self):
        """Equal Error Rate (EER)
            Equal Error Rate (EER) is the point on the ROC curve
            that corresponds to have an equal probability of miss-classifying a positive or negative sample.
            This point is obtained by intersecting the ROC curve with a diagonal of the unit square.
        """
        pass