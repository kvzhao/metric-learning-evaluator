"""ResultContainer
  @kv, lotus
"""
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd


class ResultContainer(object):
    """
      The evaluation result container handles the computation outcomes
      and save them into pandas.DataFrame.

      1. add
      2. add_event
    """
    def __init__(self):
        """
          Args:
            metrics, dict:
                Generated from ConfigParser.get_metrics()

            attributes, list of str:
                Generated from ConfigParser.get_attributes()
        """
        self._results = pd.DataFrame()
        # A buffer for storing intermediate results,
        # only show when off-line mode is used.
        self._event_buffer = pd.DataFrame()

    def add(self, attribute, metric, value, condition=None):
        """Add one result
            * create dict if key does not exist

            NOTE: threshold can not be None
        """
        self._results = self._results.append({'attribute': attribute,
                                              'metric': metric,
                                              'value': value,
                                              'condition_name': None,
                                              'condition_threshold': None}, ignore_index=True)
        if condition:
            for _cond_name, _threshold in condition.items():
                self._results = self._results.append({'attribute': attribute,
                                                      'metric': metric,
                                                      'value': value,
                                                      'condition_name': _cond_name,
                                                      'condition_threshold': _threshold}, ignore_index=True)

    def add_event(self, content):
        """
          Args:
            content: A dictionary
          Note:
        """
        self._event_buffer = self._event_buffer.append(content, ignore_index=True)

    def save(self, path):
        """Save result and events to disk"""
        os.makedirs(path, exist_ok=True)
        self._results.to_pickle(os.path.join(path, 'results.pickle'))
        self._event_buffer.to_pickle(os.path.join(path, 'events.pickle'))
        print('Save results and events into \'{}\''.format(path))

    def load(self, path):
        """load result and events from disk"""
        self._results = pd.read_pickle(os.path.join(path, 'results.pickle'))
        self._event_buffer = pd.read_pickle(os.path.join(path, 'events.pickle'))
        print('Load results and events from \'{}\''.format(path))

    @property
    def events(self):
        return self._event_buffer

    @events.setter
    def events(self, events):
        self._event_buffer = pd.DataFrame(events)

    @property
    def results(self):
        return self._results

    @property
    def flatten(self):
        dict_flatten = {}
        for row in self._results.itertuples(index=False):
            attribute = row.__getattribute__('attribute')
            metric = row.__getattribute__('metric')
            value = row.__getattribute__('value')
            condition_name = row.__getattribute__('condition_name')
            condition_threshold = row.__getattribute__('condition_threshold')
            name = '{}/{}'.format(attribute, metric)
            if condition_name and condition_threshold:
                name += '@{}={}'.format(condition_name, condition_threshold)
            dict_flatten[name] = value
        return dict_flatten

    @flatten.setter
    def flatten(self, fict_flatten):
        pass

    def clear(self):
        self._results = {}
        self._results = pd.DataFrame()
        self._event_buffer = pd.DataFrame()
        print('Clear result container.')
