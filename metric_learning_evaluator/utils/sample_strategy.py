import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import math
import numpy as np

import itertools
from random import shuffle
from collections import Counter
from collections import defaultdict

from metric_learning_evaluator.core.standard_fields import SampleStrategyStandardFields as sample_fields


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
            self._num_of_total_classes = len(self._classes)
            self._num_of_total_instances = sum(self._class_distribution.values())
            self._instance_id_hits = [0] * self._num_of_total_instances
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
            self._num_of_total_classes, self._num_of_total_instances))
        # Statistics:

    # As a most basic function
    @staticmethod
    def _sample(instance_ids,
                label_ids,
                num_of_total_classes,
                num_of_total_instances,
                class_sample_method,
                instance_sample_method,
                num_of_sampled_class,
                num_of_sampled_instance_per_class,
                maximum_of_sampled_data=None):
        """Fundamental sampling function widely used in sampler. 
          Args:
            class_sample_method: string
                sampling strategy
            instance_sample_method: string
                Only uniform sampling supported, sample_fields.uniform.
            num_of_sampled_class: Integer or String
                Number of sampled instances per class. Sanity check
            num_of_sampled_instance: Integer of String
                Number of sampled instances per class. Sanity check
            maximum_of_sampled_data:
                Integer

          Returns:
            A dict of sampled instance and label ids

          Strategy:
            * Trade-off between #of required and #of total data and if sampled data exceeds maximum
              reduce instances per class.
            * If all_instance is chosen, no maximum_of_sampled will be checked
        """
        # sanity check
        if isinstance(num_of_sampled_class, str):
            if num_of_sampled_class == sample_fields.all_class:
                num_of_sampled_class = num_of_total_classes
        elif not isinstance(num_of_sampled_class, int):
            raise TypeError('num_of_sampled_class must be integer')

        maximum_check = True
        if isinstance(num_of_sampled_instance_per_class, str):
            if num_of_sampled_instance_per_class not in [sample_fields.all_instance]:
                raise NotImplementedError('{} is not supported'.format(
                    num_of_sampled_instance_per_class))
            else:
                maximum_check = False
        elif not isinstance(num_of_sampled_instance_per_class, int):
            raise TypeError('num_of_sampled_instance_per_class must be integer')

        if maximum_check:
            if num_of_sampled_class > num_of_total_classes:
                print('Assigned #of class > provided ({}>{}), sample {} classes only.'.format(
                    num_of_sampled_class, num_of_total_classes, num_of_total_classes))
                num_of_sampled_class = num_of_total_classes
            probable_num_of_sampled_data = num_of_sampled_class * num_of_sampled_instance_per_class
            if maximum_of_sampled_data:
                upper_bound_of_sampled_data = min(num_of_sampled_class * num_of_sampled_instance_per_class,
                                                  maximum_of_sampled_data)
            else:
                if probable_num_of_sampled_data > num_of_total_instances:
                    upper_bound_of_sampled_data = num_of_total_instances
                else:
                    upper_bound_of_sampled_data = probable_num_of_sampled_data
            if probable_num_of_sampled_data > upper_bound_of_sampled_data:
                reduced_num_of_sampled_instance = math.floor(
                    upper_bound_of_sampled_data / num_of_sampled_class)
                print(
                    'Notice: Reduce number of sampled instances per class from {} to {} due to limitation.'.format(
                        num_of_sampled_instance_per_class, reduced_num_of_sampled_instance))
                num_of_sampled_instance_per_class = reduced_num_of_sampled_instance

        class_ids = list(set(label_ids))
        if class_sample_method == sample_fields.uniform:
            sampled_classes = np.random.choice(
                class_ids, num_of_sampled_class, replace=False)
        elif class_sample_method == sample_fields.all_class:
            sampled_classes = class_ids
        elif class_sample_method == sample_fields.instance_number_weighted:
            raise NotImplementedError
        elif class_sample_method == sample_fields.instance_number_inverse_weighted:
            raise NotImplementedError
        else:
            print('class sample method {} is not defined, use {} as default.'.format(
                class_sample_method, sample_fields.uniform))
            sampled_classes = np.random.choice(
                class_ids, num_of_sampled_class, replace=False)

        if not instance_sample_method in [sample_fields.uniform, sample_fields.all_instance]:
            instance_sample_method = sample_fields.uniform
            print('instance sample method {} is not defined, use {} as default.'.format(
                instance_sample_method, sample_fields.uniform))
        
        instance_group = defaultdict(list)
        for _inst_id, _class in zip(instance_ids, label_ids):
            instance_group[_class].append(_inst_id)
        sampled_instance_ids, sampled_label_ids = [], []
        for _class in sampled_classes:
            instance_ids_per_class = instance_group[_class]
            _num_instance_per_class = len(instance_ids_per_class)
            if isinstance(num_of_sampled_instance_per_class, str) and num_of_sampled_instance_per_class == sample_fields.all_instance:
                _num_sampled_instance_per_class = _num_instance_per_class
                sampled_instances = instance_ids_per_class
            else:
                _num_sampled_instance_per_class = min(_num_instance_per_class,
                                                      num_of_sampled_instance_per_class)
                if instance_sample_method == sample_fields.uniform:
                    sampled_instances = np.random.choice(instance_ids_per_class,
                                                         _num_sampled_instance_per_class,
                                                         replace=False)

            for sampled_instance in sampled_instances:
                sampled_instance_ids.append(sampled_instance)
            sampled_label_ids.extend([_class] * _num_sampled_instance_per_class)
            #self._class_histogram[_class] += _num_sampled_instance_per_class

        print('sampler: #of sampled={}'.format(len(sampled_instance_ids)))
        return {sample_fields.sampled_instance_ids: sampled_instance_ids,
                sample_fields.sampled_label_ids: sampled_label_ids}

    def sample(self,
               class_sample_method,
               instance_sample_method,
               num_of_sampled_class,
               num_of_sampled_instance_per_class,
               maximum_of_sampled_data=None):
        """Sample instance & label ids:
          Args:
            class_sample_method: string
                sampling strategy
            instance_sample_method: string
                Only uniform sampling supported, sample_fields.uniform.
            num_of_sampled_class: Integer or String
                Number of sampled instances per class. Sanity check
            num_of_sampled_instance_per_class: Integer of String
                Number of sampled instances per class. Sanity check
            maximum_of_sampled_data:
                Integer
          Returns:
            A dict of sampled instance and label ids
        """
        # TODO @kv: get return and count in histogram?
        return self._sample(instance_ids=self._instance_ids,
                            label_ids=self._label_ids,
                            num_of_total_classes=self._num_of_total_classes,
                            num_of_total_instances=self._num_of_total_instances,
                            class_sample_method=class_sample_method,
                            instance_sample_method=instance_sample_method,
                            num_of_sampled_class=num_of_sampled_class,
                            num_of_sampled_instance_per_class=num_of_sampled_instance_per_class,
                            maximum_of_sampled_data=maximum_of_sampled_data)

    # TEMPORAL IMPLEMENTATION
    def sample_pairs(self,
                     class_sample_method,
                     instance_sample_method,
                     num_of_pairs,
                     ratio_of_positive_pair,
                     ):
        """Image Pair & Sampling Strategy

          Args:

          Returns:
            A dict of pairs and label
        """

        if 2*num_of_pairs > self._num_of_total_instances:
            # reduce to the maximum
            num_of_pairs = self._num_of_total_instances // 2

        ratio_of_positive_pair = min(ratio_of_positive_pair, 1.0)

        num_of_same_pairs = math.floor(ratio_of_positive_pair * num_of_pairs)
        num_of_not_same_pairs = num_of_pairs - num_of_same_pairs
        num_of_class_coverage = num_of_same_pairs + 2 * num_of_not_same_pairs

        pairs = defaultdict(list)
        if self._num_of_total_classes < num_of_class_coverage:
            # standard sampling procedure
            num_of_class_for_same_pair = num_of_same_pairs
            num_of_class_for_not_same_pair = num_of_not_same_pairs
            sampled_class_for_same_pair = np.random.choice(self._classes, num_of_class_for_same_pair)
            sampled_class_for_not_same_pair_A = np.random.choice(self._classes, num_of_class_for_not_same_pair)
            sampled_class_for_not_same_pair_B = np.random.choice(self._classes, num_of_class_for_not_same_pair)
        else:
            # Nc is large, hard to cover by sampling
            sampled_class = np.random.choice(
                self._classes, num_of_class_coverage, replace=False)
            _1st_section_idx = num_of_same_pairs
            _2nd_section_idx = num_of_same_pairs + num_of_not_same_pairs
            sampled_class_for_same_pair = sampled_class[:_1st_section_idx]
            sampled_class_for_not_same_pair_A = sampled_class[_1st_section_idx:_2nd_section_idx]
            sampled_class_for_not_same_pair_B = sampled_class[_2nd_section_idx:]
        # push same pairs
        for _class in sampled_class_for_same_pair:
            if len(self._instance_group[_class]) < 2:
                continue
            inst_a, inst_b = np.random.choice(self._instance_group[_class], 2, replace=False)
            pairs[sample_fields.pair_A].append(inst_a)
            pairs[sample_fields.pair_B].append(inst_b)
            pairs[sample_fields.is_same].append(True)
        # push diff pairs
        for _class_a, _class_b in zip(sampled_class_for_not_same_pair_A, sampled_class_for_not_same_pair_B):
            inst_a = np.random.choice(self._instance_group[_class_a], 1, replace=False)[0]
            inst_b = np.random.choice(self._instance_group[_class_b], 1, replace=False)[0]
            is_same = _class_a == _class_b
            pairs[sample_fields.pair_A].append(inst_a)
            pairs[sample_fields.pair_B].append(inst_b)
            pairs[sample_fields.is_same].append(is_same)
        num_sampled_same_pair = sum(pairs[sample_fields.is_same])
        print('sampler: {} pairs are sampled with {} same pairs'.format(
            len(pairs[sample_fields.is_same]), num_sampled_same_pair))
        return pairs

    def _deprecated_sample_pairs(self,
                                 class_sample_method,
                                 instance_sample_method,
                                 num_of_pairs,
                                 ratio_of_positive_pair):
        """Image Pair & Sampling Strategy

          Args:

          Return:
            pairs, dict of list:

            A: {a1, a2, ..., aN} N instances in class A
            B: {b1, b2, ..., bM} M instances in class B

          Returns:
            A dict of pairs and label
        """
        pairs = defaultdict(list)
        num_instance_ids = len(self._instance_ids)
        num_label_ids = len(self._label_ids)
        num_classes = len(self._classes)

        assert num_label_ids == num_instance_ids

        # Randomly sample several data
        instance_ids = np.asarray(self._instance_ids)
        label_ids = np.asarray(self._label_ids)

        # NOTE: Hard-coded for current app
        # TODO: REMOVE THIS
        ratio_of_class = 1.0
        num_of_instance_per_class = 2
        num_sampled_classes = math.ceil(ratio_of_class * num_classes)

        sampled_classes = self._classes

        num_sampled_classes = len(sampled_classes)

        num_of_pair_counter = 0
        all_combinations = list(itertools.combinations_with_replacement(sampled_classes, 2))
        shuffle(all_combinations)

        print('num of all comb. {}'.format(len(all_combinations)))
        print('num of sampled classes {}'.format(num_sampled_classes))

        # TODO @kv: Stack to record  seen class, for per_instance and has instance shown
        class_histogram = {}
        for _class in sampled_classes:
            class_histogram[_class] = 0

        sufficient_sample = False
        for class_a, class_b in all_combinations:
            num_img_class_a = self._class_distribution[class_a]
            num_img_class_b = self._class_distribution[class_b]
            num_sampled_img_per_class = min(num_img_class_a, num_img_class_b, num_of_instance_per_class)

            # statistics
            class_histogram[class_a] += num_sampled_img_per_class
            class_histogram[class_b] += num_sampled_img_per_class

            sampled_img_id_class_a = np.random.choice(
                self._instance_group[class_a], num_sampled_img_per_class)
            sampled_img_id_class_b = np.random.choice(
                self._instance_group[class_b], num_sampled_img_per_class)

            # Add instances in pair
            for img_class_a, img_class_b in zip(sampled_img_id_class_a, sampled_img_id_class_b):
                is_same = class_a == class_b
                pairs[sample_fields.pair_A].append(img_class_a)
                pairs[sample_fields.pair_B].append(img_class_b)
                pairs[sample_fields.is_same].append(is_same)
                num_of_pair_counter += 1

            # check the strategic stop criteria
            if num_of_pair_counter > num_of_pairs:
                sufficient_sample = True
                """Find out which classes are less than requirements and sample them directly.
                """
                for _class, _counts in class_histogram.items():
                    total_num_of_instance = len(self._instance_group[_class])
                    if _counts <= num_of_instance_per_class < total_num_of_instance:
                        sufficient_sample = False
                        break
            if sufficient_sample:
                break

        print('{} pairs are generated.'.format(num_of_pair_counter))
        return pairs

    def sample_query_and_database(self,
                                  class_sample_method,
                                  instance_sample_method,
                                  num_of_db_instance_per_class,
                                  num_of_query_class,
                                  num_of_query_instance_per_class,
                                  maximum_of_sampled_data=None):
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
          TODO:
            - keyword supports: all_instance (different strategy when used in database & query)
        """
        if isinstance(num_of_query_class, str):
            if num_of_query_class == sample_fields.all_class:
                num_of_query_class = self._num_of_total_classes
        elif not isinstance(num_of_query_class, int):
            raise TypeError('num_of_query_class must be integer')

        if num_of_query_class > self._num_of_total_classes:
            num_of_query_class = self._num_of_total_classes

        # Sanity Check: Nt >= Ndb + Nq, but not really useful
        num_of_total_database_required = self._num_of_total_classes * num_of_db_instance_per_class
        num_of_total_query_required = num_of_query_class * num_of_query_instance_per_class

        """
        # If summed more than total, squeeze number of query (instances per class)
        if (num_of_total_database_required + num_of_total_query_required) > self._num_of_total_instances:
            rest_num_of_instances = self._num_of_total_instances - num_of_total_database_required
            pass
        """

        # Split instance group into database & query candidates
        sampled_query_instance_ids, sampled_query_label_ids = [], []
        sampled_database_instance_ids, sampled_database_label_ids = [], []
        sampled_query_classes = np.random.choice(self._classes, num_of_query_class, replace=False)
        required_num_of_instance_per_class = num_of_db_instance_per_class + num_of_query_instance_per_class

        # Go through all classes
        for _class in self._classes:
            # Shuffle first
            shuffle(self._instance_group[_class])
            instance_ids_of_given_class = self._instance_group[_class]

            if _class in sampled_query_classes:
                need_query = True
            else:
                need_query = False

            num_instance_given_class = len(instance_ids_of_given_class)

            if num_instance_given_class == 1:
                # can not split, skip
                print('NOTICE: label_id:{} #of instance==1, Skipped.'.format(_class))
                continue
            elif num_instance_given_class < required_num_of_instance_per_class:
                # ======== TODO: need more specific rules ==========
                if not need_query:
                    if num_instance_given_class < num_of_db_instance_per_class:
                        sampled_database_instance_ids.extend(instance_ids_of_given_class)
                        sampled_database_label_ids.extend([_class] * num_instance_given_class)
                    else:
                        _num_instance_used_for_db = min(num_instance_given_class, required_num_of_instance_per_class)
                        _sampled_instance_ids = np.random.choice(
                            instance_ids_of_given_class, _num_instance_used_for_db, replace=False)
                        sampled_database_instance_ids.extend(_sampled_instance_ids[:num_of_db_instance_per_class])
                        sampled_database_label_ids.extend([_class] * num_of_db_instance_per_class)
                else:
                    # ===== NEED QUERY ======
                    # TODO: Not a good strategy
                    if num_of_db_instance_per_class < num_instance_given_class:
                        # complete database buffer, set the rest as query
                        _reduced_num_of_query = num_instance_given_class - num_of_db_instance_per_class
                        _reduced_num_of_database = num_instance_given_class - _reduced_num_of_query
                        sampled_database_instance_ids.extend(instance_ids_of_given_class[:num_of_db_instance_per_class])
                        sampled_database_label_ids.extend([_class] * num_of_db_instance_per_class)
                        sampled_query_instance_ids.extend(instance_ids_of_given_class[num_of_db_instance_per_class:])
                        sampled_query_label_ids.extend([_class] * _reduced_num_of_query)
                    elif num_of_query_instance_per_class < num_instance_given_class:
                        # complete query buffer, set the rest as database
                        _reduced_num_of_database = num_instance_given_class - num_of_query_instance_per_class
                        _reduced_num_of_query = num_instance_given_class - _reduced_num_of_database
                        sampled_query_instance_ids.extend(instance_ids_of_given_class[:num_of_query_instance_per_class])
                        sampled_query_label_ids.extend([_class] * num_of_query_instance_per_class)
                        sampled_database_instance_ids.extend(instance_ids_of_given_class[num_of_query_instance_per_class:])
                        sampled_database_label_ids.extend([_class] * _reduced_num_of_database)
                    else:
                        # split into half
                        _reduced_num_of_query = int(num_instance_given_class / 2)
                        _reduced_num_of_database = num_instance_given_class - _reduced_num_of_query
                        sampled_database_instance_ids.extend(instance_ids_of_given_class[:_reduced_num_of_query])
                        sampled_database_label_ids.extend([_class] * _reduced_num_of_query)
                        sampled_query_instance_ids.extend(instance_ids_of_given_class[_reduced_num_of_query:])
                        sampled_query_label_ids.extend([_class] * _reduced_num_of_database)
                    print('''NOTICE: label_id:{} Required instances(#={}) is more than acquired(#={}), '''
                          '''Reduced (#of query={}, #of db={}).'''.format(
                        _class, required_num_of_instance_per_class, num_instance_given_class,
                        _reduced_num_of_query, _reduced_num_of_database))
            else:
                _sampled_instance_ids = np.random.choice(
                    instance_ids_of_given_class, required_num_of_instance_per_class, replace=False)
                sampled_database_instance_ids.extend(_sampled_instance_ids[:num_of_db_instance_per_class])
                sampled_database_label_ids.extend([_class] * num_of_db_instance_per_class)

                if need_query:
                    sampled_query_instance_ids.extend(_sampled_instance_ids[num_of_db_instance_per_class:])
                    sampled_query_label_ids.extend([_class] * num_of_query_instance_per_class)

        # Pass candidates into _sample function
        print('sampler: #of query={}, #of database={}'.format(
            len(sampled_query_instance_ids), len(sampled_database_instance_ids)))
        return {
            sample_fields.database_instance_ids: sampled_database_instance_ids,
            sample_fields.database_label_ids: sampled_database_label_ids,
            sample_fields.query_instance_ids: sampled_query_instance_ids,
            sample_fields.query_label_ids: sampled_query_label_ids,
        }

    def _deprecated_sample_query_and_database(self,
                                              class_sample_method,
                                              instance_sample_method,
                                              num_of_db_instance,
                                              num_of_query_class,
                                              num_of_query_instance_per_class,
                                              maximum_of_sampled_data):
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
            sampled_db, sampled_query = self.deprecated_class_uniform_sampler(num_of_query_class,
                                                                              num_of_db_instance,
                                                                              num_of_query_instance_per_class,
                                                                              maximum_of_sampled_data=maximum_of_sampled_data)
        if class_sample_method == sample_fields.all_class:
            sampled_db, sampled_query = self.deprecated_class_take_all(num_of_db_instance,
                                                                       num_of_query_instance_per_class,
                                                                       instance_sample_method)
        elif class_sample_method == sample_fields.instance_number_weighted:
            raise NotImplementedError
        elif class_sample_method == sample_fields.instance_number_inverse_weighted:
            raise NotImplementedError
        else:
            print('class sample method {} is not defined, use {} as default.'.format(
                class_sample_method, sample_fields.uniform))
            sampled_db, sampled_query = self.deprecated_class_uniform_sampler(num_of_query_class,
                                                                              num_of_db_instance,
                                                                              num_of_query_instance_per_class,
                                                                              maximum_of_sampled_data=maximum_of_sampled_data)

        return {
            sample_fields.db_instance_ids: sampled_db[sample_fields.sampled_instance_ids],
            sample_fields.db_label_ids: sampled_db[sample_fields.sampled_label_ids],
            sample_fields.query_instance_ids: sampled_query[sample_fields.sampled_instance_ids],
            sample_fields.query_label_ids: sampled_query[sample_fields.sampled_label_ids]
        }

    def deprecated_class_take_all(self,
                                  num_of_db_instance_per_class,
                                  num_of_query_instance_per_class,
                                  instance_sample_method):
        """
          Args:
            num_of_db_instance_per_class:
                Integer which indicates number of given database features.
        """

        db_instances = []
        db_labels = []
        query_instances = []
        query_labels = []

        sampled_classes = self._classes

        for _class in sampled_classes:
            instance_ids_per_class = self._instance_group[_class]
            num_instance_per_class = len(instance_ids_per_class)
            if num_instance_per_class==1:
                print('Notice: class_id: {} num_instance_per_class==1, Skipped.'.format(_class))
                continue
            
            if num_of_db_instance_per_class >= num_instance_per_class:
                num_of_db_instance_per_class = math.floor(num_instance_per_class * 0.5)
                print('Notice: class_id: {} has only {} instances, reduce db instances to {}'.format(
                    _class, num_instance_per_class, num_of_db_instance_per_class))

            # Sample per class
            shuffle(instance_ids_per_class)

            db_instances_per_class = instance_ids_per_class[:num_of_db_instance_per_class]
            query_instances_per_class = instance_ids_per_class[num_of_db_instance_per_class:]
            num_query_instances = num_instance_per_class - num_of_db_instance_per_class
            # NOTE: Sampling here.
            if instance_sample_method == sample_fields.all_instance:
                sampled_query_instances_per_class = query_instances_per_class
                num_sampled_query_instances = num_query_instances
            elif instance_sample_method == sample_fields.uniform:
                num_sampled_query_instances = min(num_query_instances, num_of_query_instance_per_class)
                sampled_query_instances_per_class = np.random.choice(query_instances_per_class,
                                                                     num_sampled_query_instances,
                                                                     replace=False)
            else:
                print('instance sample method {} is not defined, use {} as default.'.format(
                instance_sample_method, sample_fields.all_instance))
                sampled_query_instances_per_class = query_instances_per_class
                num_sampled_query_instances = num_query_instances

            for instance in db_instances_per_class:
                db_instances.append(instance)
            db_labels.extend([_class] * num_of_db_instance_per_class)
            for instance in sampled_query_instances_per_class:
                query_instances.append(instance)
            query_labels.extend([_class] * num_sampled_query_instances)

            self._class_histogram[_class] += num_instance_per_class
        sampled_db = {
            sample_fields.sampled_instance_ids: db_instances,
            sample_fields.sampled_label_ids: db_labels
        }
        sampled_query = {
            sample_fields.sampled_instance_ids: query_instances,
            sample_fields.sampled_label_ids: query_labels
        }
        return sampled_db, sampled_query

    # TODO @kv: This must be deprecated.
    def deprecated_class_uniform_sampler(self,
                                         num_of_sampled_class,
                                         num_of_db_instance_per_class,
                                         num_of_query_instance_per_class,
                                         maximum_of_sampled_data=None):
        num_of_sampled_instance = num_of_db_instance_per_class + num_of_query_instance_per_class
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
        self._num_of_total_classes = None
        self._num_of_total_instances = None
        self._classes = None
        self._num_of_instance_per_class_mean = None
        self._num_of_instance_per_class_std = None
        self._num_of_instance_per_class_max = None
        self._num_of_instance_per_class_min = None
