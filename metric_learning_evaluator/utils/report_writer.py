
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import collections
import numpy as np
from metric_learning_evaluator.data_tools.result_container import ResultContainer
from pytablewriter import MarkdownTableWriter


class ReportWriter(object):
    def __init__(self, container):
        self._container = container

    @property
    def overall_report(self):
        writer = MarkdownTableWriter()
        writer.title = 'overall'
        flattens = self._container.flatten
        writer.headers = ['metric', 'value']
        values = []
        for k, v in flattens.items():
            values.append([k, v])
        writer.value_matrix = values
        return writer.dumps()

    @property
    def event_report(self):
        # TODO: Use pandas dataframe's feature
        _all_reports = ''
        all_events = collections.OrderedDict(sorted(
            self._container.events.items(), key=lambda t: len(t[0])))
        for attr, event_list in all_events.items():
            writer = MarkdownTableWriter()
            writer.title = attr
            if len(event_list) == 0:
                continue
            else:
                keys = list(event_list[0].keys())
            values = []
            writer.headers = keys
            for event in event_list:
                line = []
                for k in keys:
                    event_val = event[k]
                    if isinstance(event_val, (np.ndarray, np.generic)):
                        event_val = event_val.tolist()
                    if isinstance(event_val, int):
                        line.append(event_val)
                    elif isinstance(event_val, float):
                        line.append(event_val)
                    elif isinstance(event_val, list):
                        if all(isinstance(n, int) for n in event_val):
                            line.append(', '.join(str(v) for v in event_val))
                        elif all(isinstance(n, float) for n in event_val):
                            line.append(', '.join('{0:.5f}'.format(v) for v in event_val))
                values.append(line)
            writer.value_matrix = values
            _all_reports += writer.dumps()
        return _all_reports
