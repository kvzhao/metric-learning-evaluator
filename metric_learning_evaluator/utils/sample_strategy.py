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
    instance_number_weighted = 'instance_number_weighted'
    instance_number_inverse_weighted = 'instance_number_inverse_weighted'

    # sampling options
    sample_ratio = 'sample_ratio'

    ratio_of_instance = 'ratio_of_instance'
    ratio_of_sampled_class = 'ratio_of_sampled_class'
    ratio_of_instance_per_class = 'ratio_of_instance_per_class'

    num_of_sampled_class = 'num_of_sampled_class'
    num_of_sampled_instance_per_class = 'num_of_sampled_instance_per_class'

    num_of_db_instance = 'num_of_db_instance'
    num_of_query_class = 'num_of_query_class'
    num_of_query_instance_per_class = 'num_of_query_instance_per_class'
    maximum_of_sampled_data = 'maximum_of_sampled_data'

    # pair
    is_same = 'is_same'
    pair_A = 'pair_A'
    pair_B = 'pair_B'

    # ranking
    sampled_instance_ids = 'sampled_instance_ids'
    sampled_label_ids = 'sampled_label_ids'
    query_instance_ids = 'query_instance_ids'
    query_label_ids = 'query_label_ids'
    db_instance_ids = 'db_instance_ids'
    db_label_ids = 'db_label_ids'


sample_fields = SampleStrategyStandardFields


class SampleStrategy(object):
    """Class Sampling Strategy
        The object manages sampling strategy between class and number of instances.

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
            - instance number weighted
            - inverse instance number weighted
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
        self.verbose()

    def _setup(self):
        if not (self._label_ids is None or self._instance_ids is None):
            self._class_distribution = Counter(self._label_ids)
            self._classes = list(self._class_distribution.keys())
            self._num_of_class = len(self._classes)
            self._num_of_instances = sum(self._class_distribution.values())
            self._class_histogram = {}

            self._instance2class = {}
            self._instance_group = defaultdict(list)

            for _instance, _class in zip(self._instance_ids, self._label_ids):
                self._instance2class[_instance] = _class
                self._instance_group[_class].append(_instance)

            for _class, _count in self._class_distribution.items():
                self._class_histogram[_class] = 0

            per_class_numbers = list(self._class_distribution.values())
            self._num_of_instance_per_class_mean = np.mean(per_class_numbers)
            self._num_of_instance_per_class_std = np.std(per_class_numbers)
            self._num_of_instance_per_class_max = np.max(per_class_numbers)
            self._num_of_instance_per_class_min = np.min(per_class_numbers)

    def verbose(self):
        # Show information about distribution
        print('sampler: {} classes with {} instances'.format(
            self._num_of_class, self._num_of_instances))

    def _sample(self,
                class_sample_method,
                instance_sample_method,
                num_of_sampled_class,
                num_of_sampled_instance,
                maximum_of_sampled_data=None,
                ):
        """
          Args:
            class_sample_method
            instance_sample_method
            num_of_sampled_instance
            num_of_sampled_class
            maximum_of_sampled_data

          Returns:
            A dict of class and instances: 

          Strategy:
            Trade-off between num_of_class and num_of_data, if sampled data exceeds maximum
            reduce instances per class.
        """
        probable_num_of_sampled_data = num_of_sampled_class * num_of_sampled_instance
        if maximum_of_sampled_data:
            upper_bound_of_sampled_data = min(num_of_sampled_class * num_of_sampled_instance,
                                              maximum_of_sampled_data)
        else:
            upper_bound_of_sampled_data = probable_num_of_sampled_data
        if probable_num_of_sampled_data > upper_bound_of_sampled_data:
            reduced_num_of_sampled_instance = math.floor(
                upper_bound_of_sampled_data / num_of_sampled_class)
            print(
                'Notice: Reduce number of sampled instances per class from {} to {} due to limitation.'.format(
                    num_of_sampled_instance, reduced_num_of_sampled_instance))
            num_of_sampled_instance = reduced_num_of_sampled_instance

        if class_sample_method == sample_fields.uniform:
            sampled_classes = np.random.choice(
                self._classes, num_of_sampled_class, replace=False)
        elif class_sample_method == sample_fields.instance_number_weighted:
            raise NotImplementedError
        elif class_sample_method == sample_fields.instance_number_inverse_weighted:
            raise NotImplementedError
        else:
            print('class sample method {} is not defined, use {} as default.'.format(
                class_sample_method, sample_fields.uniform))
            sampled_classes = np.random.choice(
                self._classes, num_of_sampled_class, replace=False)

        if not instance_sample_method in [sample_fields.uniform]:
            instance_sample_method = sample_fields.uniform
            print('instance sample method {} is not defined, use {} as default.'.format(
                instance_sample_method, sample_fields.uniform))

        instances = []
        labels = []
        for _class in sampled_classes:
            instance_ids_per_class = self._instance_group[_class]
            num_instance_per_class = len(instance_ids_per_class)
            num_sampled_instance_per_class = min(num_instance_per_class,
                                                 num_of_sampled_instance)

            if instance_sample_method == sample_fields.uniform:
                sampled_instances = np.random.choice(instance_ids_per_class,
                                                     num_sampled_instance_per_class,
                                                     replace=False)
            else:
                raise NotImplementedError

            for sampled_instance in sampled_instances:
                instances.append(sampled_instance)
            labels.extend([_class] * num_sampled_instance_per_class)
            self._class_histogram[_class] += num_sampled_instance_per_class

        return {sample_fields.sampled_instance_ids: instances,
                sample_fields.sampled_label_ids: labels}

    def sample_pairs(self,
                     class_sample_method,
                     instance_sample_method,
                     maximum_of_sampled_data):
        """
          Returns:
            A dict of pairs and label
        """
        pass

    def sample_query_and_database(self,
                                  class_sample_method,
                                  instance_sample_method,
                                  num_of_db_instance,
                                  num_of_query_class,
                                  num_of_query_instance_per_class,
                                  maximum_of_sampled_data,
                                  ):
        """
          Args:
            class_sample_method:
            instance_sample_method:
            num_of_db_instance:
            num_of_query_class:
            num_of_query_instance_per_class:
            maximum_of_sampled_data:
          Returns:
            Dict of db instances, labels and query instances, labels.
        """

        # NOTE @dennis.liu: Split _sample method into specific sample function
        if class_sample_method == sample_fields.uniform:
            sampled_db, sampled_query = self.class_uniform_sampler(num_of_query_class,
                                                                   num_of_db_instance,
                                                                   num_of_query_instance_per_class,
                                                                   maximum_of_sampled_data=maximum_of_sampled_data)
        elif class_sample_method == sample_fields.instance_number_weighted:
            raise NotImplementedError
        elif class_sample_method == sample_fields.instance_number_inverse_weighted:
            raise NotImplementedError
        else:
            print('class sample method {} is not defined, use {} as default.'.format(
                class_sample_method, sample_fields.uniform))
            sampled_db, sampled_query = self.class_uniform_sampler(num_of_query_class,
                                                                   num_of_db_instance,
                                                                   num_of_query_instance_per_class,
                                                                   maximum_of_sampled_data=maximum_of_sampled_data)

        return {
            sample_fields.db_instance_ids: sampled_db[sample_fields.sampled_instance_ids],
            sample_fields.db_label_ids: sampled_db[sample_fields.sampled_label_ids],
            sample_fields.query_instance_ids: sampled_query[sample_fields.sampled_instance_ids],
            sample_fields.query_label_ids: sampled_query[sample_fields.sampled_label_ids]
        }

    def class_uniform_sampler(self,
                              num_of_sampled_class,
                              num_of_db_instance_per_class,
                              num_of_query_instance_per_class,
                              maximum_of_sampled_data=None):
        num_of_sampled_instance = num_of_db_instance_per_class + \
                                  num_of_query_instance_per_class
        # probable_num_of_sampled_data = num_of_sampled_class * num_of_sampled_instance
        # if maximum_of_sampled_data:
        #     upper_bound_of_sampled_data = min(num_of_sampled_class * num_of_sampled_instance,
        #                                       maximum_of_sampled_data)
        # else:
        #     upper_bound_of_sampled_data = probable_num_of_sampled_data
        #
        # if probable_num_of_sampled_data > upper_bound_of_sampled_data:
        #     reduced_num_of_sampled_instance = math.floor(
        #         upper_bound_of_sampled_data / num_of_sampled_class)
        #     print(
        #         'Notice: Reduce number of sampled instances per class from {} to {} due to limitation.'.format(
        #             num_of_sampled_instance, reduced_num_of_sampled_instance))
        #     num_of_sampled_instance = reduced_num_of_sampled_instance
        db_instances = []
        db_labels = []
        query_instances = []
        query_labels = []

        # Sample id of class(label)
        if len(self._classes) < num_of_sampled_class:
            print('Warning: num_of_sampled_class > num_class({})" \
                    "set num_of_sampled_class = {}'.format(
                num_of_sampled_class, len(self._classes), len(self._classes)))
            num_of_sampled_class = len(self._classes)

        sampled_classes = np.random.choice(
            self._classes, num_of_sampled_class, replace=False)

        for _class in sampled_classes:
            instance_ids_per_class = self._instance_group[_class]
            num_instance_per_class = len(instance_ids_per_class)
            if num_instance_per_class <= 1:
                print('Warning: class_id: {} num_instance_per_class==1, Skipped...'.format(_class))
                continue

            # Constraint number of samples(db and query) per class
            num_sampled_instance_per_class = min(num_instance_per_class,
                                                 num_of_sampled_instance)



            # Sample per class
            sampled_instances = np.random.choice(instance_ids_per_class,
                                                 num_sampled_instance_per_class,
                                                 replace=False)
            if len(sampled_instances) <= 1:
                print('Warning: class_id: {} sampled_instances <= 1, Skipped...'.format(_class))
                continue

            num_query_instances = num_of_query_instance_per_class
            num_db_instances = min(len(sampled_instances) - num_query_instances,
                                   num_of_db_instance_per_class)

            if num_db_instances <= 0:
                print('Warning: num_query_instances > num_sampled_instances, ' \
                      'Reduce number of query instances(num_query_instances) into 1.')
                num_query_instances = 1
                num_db_instances = len(sampled_instances) - num_query_instances

            db_instances_per_class = sampled_instances[:num_db_instances]
            query_instances_per_class = sampled_instances[num_db_instances:
                                                          num_db_instances + num_query_instances]

            for instance in db_instances_per_class:
                db_instances.append(instance)
            db_labels.extend([_class] * num_db_instances)
            for instance in query_instances_per_class:
                query_instances.append(instance)
            query_labels.extend([_class] * num_query_instances)

            self._class_histogram[_class] += num_sampled_instance_per_class
        sampled_db = {
            sample_fields.sampled_instance_ids: db_instances,
            sample_fields.sampled_label_ids: db_labels
        }
        sampled_query = {
            sample_fields.sampled_instance_ids: query_instances,
            sample_fields.sampled_label_ids: query_labels
        }
        return sampled_db, sampled_query

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
