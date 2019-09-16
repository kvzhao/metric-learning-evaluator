"""
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from collections import defaultdict

from metric_learning_evaluator.core.standard_fields import AttributeTableStandardFields as attrtable_fields


class AttributeTable(object):
    """
      Use dataframe
      This is more like attribute container
    """

    def __init__(self, table_path=None):
        self._attribute_df = None
        self._table_path = None
        self._init_internals()

        if table_path is not None:
            self._table_path = table_path
            self.load(self._table_path)

    def _init_internals(self):
        self._instance_ids = []
        self._attribute_buffer = []
        self._attribute_types = set()

    def createIndex(self, recreate=False):
        """Create Pandas DataFrame as Index
            NOTE: createIndex doesn't check whether dataframe and buffer are the same or not.
        """
        if self.has_index and recreate:
            print('NOTICE: internal pandas DataFrame is created already')
            return
        dict_attributes = defaultdict(list)
        for instance_id, attr_dict in zip(self._instance_ids, self._attribute_buffer):
            dict_attributes[attrtable_fields.instance_id].append(instance_id)
            for attr_type in self.attribute_types:
                if attr_type == attrtable_fields.instance_id:
                    continue
                attr = attr_dict.get(attr_type, None)
                dict_attributes[attr_type].append(attr)
        self._attribute_df = pd.DataFrame(dict_attributes)

    @property
    def has_index(self):
        return self._attribute_df is not None

    @property
    def attributes(self):
        return self._attribute_buffer

    @property
    def DataFrame(self):
        if not self.has_index:
            self.createIndex()
        return self._attribute_df

    @property
    def attribute_types(self):
        return list(self._attribute_types)

    def add(self, instance_id, attributes=None):
        """
          Args:
            instance_id: an integer
            attributes: a dictionary
              e.g.
                instace_id=10, {source: se7en, color: red, seen_or_unseen: unseen}
        """
        try:
            instance_id = int(instance_id)
        except:
            raise TypeError("The instance id has wrong type")
        self._instance_ids.append(instance_id)
        if attributes is None:
            attributes = {}
        self._attribute_buffer.append(attributes)
        for k, _ in attributes.items():
            self._attribute_types.add(k)

    def _query(self, key, value):
        """handy function for kv query"""
        command = '{key}==\'{value}\''.format(key, value)
        return self.DataFrame.query(command)[attrtable_fields.instance_id].tolist()

    def save(self, path):
        """
          Args:
            path: String, path to the csv file.
          NOTE: operation would not be executed if index not created.
        """
        if not self.has_index:
            return
        self._attribute_df.to_csv(path)

    def load(self, path):
        """
          Args:
            path: String, path to the given csv file.
          NOTE:
            The function would clear existing internals and load all from file.
          Raise:
            FileNotFoundError
        """
        if not os.path.exists(path):
            raise FileNotFoundError('{} does not exists'.format(path))
        if self.has_index:
            print('NOTICE:')
        self._attribute_df = pd.read_csv(path, index_col=0)

    def clear(self):
        self._init_internals()
        self._attribute_df = None
