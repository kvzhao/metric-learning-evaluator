"""
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from abc import ABCMeta
from abc import abstractmethod
import collections
from collections import defaultdict

import numpy as np



class ResultContainer(object):
    """
      The evaluation result container handles the computation outcomes
      and save them into the unified data structure.

      NOTE:
        Structure of the result_container:
    """

    def __init__(self, metrics, attributes):
        """
          Args:
            metrics, dict:
                Generated from ConfigParser.get_metrics()

            attributes, list of str:
                Generated from ConfigParser.get_attributes()

        """
        self._results = {}
        # allocate the dictionary
        if not isinstance(attributes, list):
            attributes = [attributes]

        for attr in attributes:
            self._results[attr] = {}
            for metric, _ in metrics.items():
                self._results[attr][metric] = {}

    def __repr__(self):
        """
            Print the Result in structure.
                maybe markdown.
        """
        result_string = ''
        for _attr_name, _metirc in self._results.items():
            for _metric_name, _threshold in _metirc.items():
                for _thres, _value in _threshold.items():
                    if not _value:
                        continue
                    result_string += '{}-{}@{}: {}\n'.format(
                        _metric_name, _attr_name, _thres, _value)
        return result_string

    def add(self, attribute, metric, threshold, value):
        """Add one result
            * create dict if key does not exist
        """
        if not attribute in self._results:
            self._results[attribute] = {}
        if not metric in self._results[attribute]:
            self._results[attribute][metric] = {}
        self._results[attribute][metric][threshold] = value

    @property
    def results(self):
        # TODO: Do not return empty dict
        dict_outcome = {}
        for _attr_name, _metirc in self._results.items():
            dict_outcome[_attr_name] = {}
            for _metric_name, _threshold in _metirc.items():
                if not _threshold:
                    continue
                dict_outcome[_attr_name][_metric_name] = {}
                for _thres, _value in _threshold.items():
                    if not _value:
                        continue
                    dict_outcome[_attr_name][_metric_name][_thres] = _value
        return dict_outcome

    @property
    def flatten(self):
        dict_flatten = {}
        for _attr_name, _metirc in self._results.items():
            for _metric_name, _threshold in _metirc.items():
                for _thres, _value in _threshold.items():
                    if not _value:
                        continue
                    _name = '{}-{}@{}'.format(_attr_name, _metric_name, _thres)
                    dict_flatten[_name] = _value
        return dict_flatten

    def clear(self):
        self._results = {}
        print ('Clear result container.')