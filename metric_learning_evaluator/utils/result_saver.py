
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
from metric_learning_evaluator.data_tools.result_container import ResultContainer

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class ResultSaver(object):

    def __init__(self, container=None):
        self._container = container
        if self._container is None:
            self._container = ResultContainer()

    def save_overall(self, path):
        if self._container is None:
            return
        overall = self._container.flatten
        path += '.json'
        try:
            with open(path, 'w') as fp:
                json.dump(overall, fp, cls=NumpyEncoder)
        except:
            print('{} can not open'.format(path))

    def save_event(self, path):
        if self._container is None:
            return
        # dict of list of dict
        all_events = self._container.events
        for eval_name, event_list in all_events.items():
            path = path + '_' + eval_name + '.json'
            with open(path, 'w') as fp:
                json.dump(event_list, fp)

    def load(self, path):
        """
          Args:
            path: String, path to given json file.
          Return:
            list of dict
        """
        try:
            with open(path, 'r') as fp:
                json_load = json.load(fp)
            return json_load
        except Exception as e:
            print('{} loading fails'.format(path))
            return []

    @property
    def result_container(self):
        self._container