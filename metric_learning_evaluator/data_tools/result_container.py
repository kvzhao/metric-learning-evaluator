"""
"""
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class ResultContainerStandardFields:
    pass


class ResultContainer(object):
    """
      The evaluation result container handles the computation outcomes
      and save them into the unified data structure.

      NOTE:
        Structure of the result_container:
          - evaluation_name (outer structure)
            - attribute_name: string
              - metric_name: string
                  - value: float
                    - condition: dict (e.g @distance=0.5)
    """

    def __init__(self):
        """
          Args:
            metrics, dict:
                Generated from ConfigParser.get_metrics()

            attributes, list of str:
                Generated from ConfigParser.get_attributes()

        """
        self._results = {}

    def add(self, attribute, metric, value, condition=None):
        """Add one result
            * create dict if key does not exist

            NOTE: threshold can be None
        """
        if attribute not in self._results:
            self._results[attribute] = {}
        if metric not in self._results[attribute]:
            self._results[attribute][metric] = {}

        if condition is None:
            self._results[attribute][metric][''] = value
        elif isinstance(condition, dict):
            _cond_key = ''
            for _cond_name, _threshold in condition.items():
                _cond_key += '@{}={}'.format(_cond_name, _threshold)
            self._results[attribute][metric][_cond_key] = value
        elif isinstance(condition, str):
            _cond_key = condition
            self._results[attribute][metric][_cond_key] = value

    @property
    def results(self):
        # TODO: Do not return empty dict
        dict_outcome = {}
        for _attr_name, _metric in self._results.items():
            dict_outcome[_attr_name] = {}
            for _metric_name, _content in _metric.items():
                if not _content:
                    continue
                dict_outcome[_attr_name][_metric_name] = _content
        return dict_outcome

    @property
    def flatten(self):
        dict_flatten = {}
        for _attr_name, _metric in self._results.items():
            for _metric_name, _content in _metric.items():
                if not _content:
                    continue
                for _cond_key, _value in _content.items():
                    if _cond_key == '':
                        _name = '{}/{}'.format(_attr_name, _metric_name)
                    else:
                        _name = '{}/{}{}'.format(_attr_name, _metric_name, _cond_key)
                    dict_flatten[_name] = _value

        return dict_flatten

    def clear(self):
        self._results = {}
        print('Clear result container.')
