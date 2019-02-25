import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import math
import numpy as np

import itertools
from collections import Counter
from collections import defaultdict

from pprint import pprint

class SampleStrategyStandardFields:

    # sampling
    sample_method = 'sample_method'
    class_sample_method = 'class_sample_method'
    instance_sample_method = 'instance_sample_method'

    # mothods
    uniform = 'uniform'
    instance_amount_weighted = 'instance_amount_weighted'
    instance_amount_inverse_weighted = 'instance_amount_inverse_weighted'

    sample_ratio = 'sample_ratio'
    ratio_of_instance = 'ratio_of_instance'
    ratio_of_sampled_class = 'ratio_of_sampled_class'
    ratio_of_instance_per_class = 'ratio_of_instance_per_class'
    num_of_sampled_class = 'num_of_sampled_class'
    num_of_sampled_instance_per_class = 'num_of_sampled_instance_per_class'
    maximum_of_sampled_data = 'maximum_of_sampled_data'

    # pair condition
    is_same = 'is_same'

sample_fields = SampleStrategyStandardFields

class SampleStrategy(object):
    """Class Sampling Strategy
        The object manages sampling strategy between class and amount of instances.

      instance_ids (intra-class)
        ^
        |
        |               ^
        |             ^   ^
        |    ^      ^       ^
        |  ^   ^ ^ ^         ^       ^
        | ^                    ^ ^ ^ 
        - - - - - - - - - - - - - - - - -> label_ids (inter-class)


      Sample methods:
        - class-wise sampling
            - uniform
            - instance amount weighted
            - inverse instance amount weighted
        - instance-wise sampling
            - uniform
            - (TODO @kv) instance-wise attribute (intra-variation) 
      
      Conditions and Constraints:
        - ratio of sampling classes
        - number of instances per class
        - maximum number of sampled data
    """

    def __init__(self, instance_ids, label_ids):
        """
          Args:
            instance_ids: list of instance id with type int or str
            label_ids: list of label id with type int
            Both instance and label ids are in same order.
        """
        self._instance_ids = instance_ids
        self._label_ids = label_ids
        self._setup()

    def _setup(self):
        if self._label_ids and self._instance_ids:
            self._class_distribution = Counter(self._label_ids)
            self._classes = list(self._class_distribution.keys())
            self._num_of_class = len(self._classes)
            self._class_histogram = {}

            self._instance2class = {}
            self._instance_group = defaultdict(list)

            for _instance, _class in zip(self._instance_ids, self._label_ids):
                self._instance2class[_instance] = _class
                self._instance_group[_class].append(_instance)

            for _class, _count in self._class_distribution.items():
                self._class_histogram[_class] = 0

            per_class_amounts = list(self._class_distribution.values())
            self._num_of_instance_per_class_mean = np.mean(per_class_amounts)
            self._num_of_instance_per_class_std = np.std(per_class_amounts)
            self._num_of_instance_per_class_max = np.max(per_class_amounts)
            self._num_of_instance_per_class_min = np.min(per_class_amounts)


    def verbose(self):
        # Show information about distribution
        pass

    def _sample(self,
                class_sample_method,
                instance_sample_method,
                ratio_of_sampled_class,
                num_of_sampled_class,
                num_of_sampled_instance,
                maximum_of_sampled_data,
                ):
        """
          Args:
            class_sample_method
            instance_sample_method
            ratio_of_sampled_class
            num_of_sampled_instance
            maximum_of_sampled_data

          Returns:
            A dict of class and instances: 
            A tuple of (instance_ids, label_ids)

          Strategy:
            Trade-off between num_of_class and max num_of_data
        """
        pass

    def sample_pairs(self):
        """
          Returns:
            A dict of pairs and label
        """
        pass

    def sample_queries(self,
                       class_sample_method,
                       instance_sample_method,
                       ratio_of_sampled_class,
                       num_of_sampled_class,
                       num_of_sampled_instance,
                       maximum_of_sampled_data,
                      ):
        """
          Args:
            class_sample_method
            instance_sample_method
            ratio_of_sampled_class
            num_of_sampled_instance
            maximum_of_sampled_data

          Returns:
            A dict of class and instances

          Strategy:
            Trade-off between num_of_class and max num_of_data
        """

        if ratio_of_sampled_class == 1.0:
            num_of_sampled_class = self._num_of_class
        else:
            num_of_sampled_class = min(self._num_of_class,
                math.ceil(self._num_of_class * ratio_of_sampled_class))

        """
        """
        probable_num_of_sampled_instances = num_of_sampled_class * num_of_sampled_instance
        upper_bound_of_sampled_instances = min(num_of_sampled_class * num_of_sampled_instance,
                                               maximum_of_sampled_data)
        
        sampled_instance_counter = 0

        if class_sample_method == sample_fields.uniform:
            sampled_classes = np.random.choice(self._classes, num_of_sampled_class)
        elif class_sample_method == sample_fields.instance_amount_weighted:
            pass
        elif class_sample_method == sample_fields.instance_amount_inverse_weighted:
            pass
        else:
            print ('class sample method {} is not define, use uniform as default.'.format(class_sample_method))
            sampled_classes = np.random.choice(self._classes, num_of_sampled_class)

        for _class in sampled_classes:
            instance_ids_per_class = self._instance_group[_class]

            num_instance_per_class = len(instance_ids_per_class)
            num_sampled_instance_per_class = min(num_instance_per_class,
                                                 num_of_sampled_instance)

            print (num_sampled_instance_per_class)
            if instance_sample_method == sample_fields.uniform:
                sampled_instances = np.random.choice(instance_ids_per_class, num_sampled_instance_per_class)
            print (sampled_instances)


    def clear(self):
        self._class_distribution = None
        self._class_histogram = None
        self._instance2class = None
        self._instance_group = None
        self._num_of_class = None
        self._classes = None
        self._num_of_instance_per_class_mean = None
        self._num_of_instance_per_class_std = None
        self._num_of_instance_per_class_max = None
        self._num_of_instance_per_class_min = None