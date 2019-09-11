import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import csv
from metric_learning_evaluator.core.standard_fields import AttributeTableStandardFields as fields

"""
"""
MUST_BE_IGNORED = ['',
                   'image_path',
                   'filename_string',
                   'label_id',
                   'label_ids',
                   'label_name',
                   'label_names',
                   'instance_id',
                   'instance_ids'
                   'width',
                   'height']


class CsvReader(object):
    def __init__(self,
                 database_config,
                 ignore_keys=[]):
        """CsvReader
          The reader of csv file

          Args:
            database_config:
                A dictionary with keys: `path`
            ignore_keys:
                A list of string, denote ignorance of key strings
        """
        path = database_config.get('path', None)
        self.dict_attributes = {}
        self._counter = 0
        self._ignore_keys = ignore_keys + MUST_BE_IGNORED

        if path is None:
            raise FileNotFoundError("path is required")

        # TODO: store all csv information
        if os.path.exists(path):
            with open(path, newline='', encoding='utf-8') as csv_file:
                csv_rows = csv.DictReader(csv_file)
                for row in csv_rows:
                    inst_id = row.get(fields.instance_id, self._counter)
                    for ignore in self._ignore_keys:
                        row.pop(ignore, None)
                    self.dict_attributes[inst_id] = row
                    self._counter += 1
        else:
            raise FileNotFoundError("Given database path: {} not found".format(path))

    def query_attributes_by_instance_id(self, instance_id):
        """
          Args:
            instance_id: integer
          Return
            attr_dict: dict, empty if no attributes exist
        """
        instance_id = str(instance_id)
        if instance_id in self.dict_attributes:
            return self.dict_attributes[instance_id]
        return {}

    # TODO:
    def query_info_by_instance_id(self, instance_id):
        pass

    @property
    def instance_ids(self):
        return list(self.dict_attributes.keys())
