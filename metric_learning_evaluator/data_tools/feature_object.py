"""
  Feature Data Reader
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import re
import inspect
import numpy as np
from abc import ABCMeta
from abc import abstractmethod
from metric_learning_evaluator.core.standard_fields import FeatureObjectStandardFields as fields

# TODO: @kv add bounding box

class FeatureObjectBase(object):

    def __init__(self):

        self._array_names = [
            fields.embeddings,
            fields.probabilities,
            fields.label_ids,
            fields.label_names,
            fields.instance_ids,
            fields.filename_strings
        ]
        self._array_name_map = {
            fields.embeddings: None,
            fields.probabilities: None,
            fields.label_ids: None,
            fields.label_names: None,
            fields.instance_ids: None,
            fields.filename_strings: None,
        }

    def __repr__(self):
        res = ''
        for _name, _arr in self._array_name_map.items():
            if not _arr is None:
                res += '{}: {}\n'.format(_name, _arr.shape)
        return res

    def _check_numpy_arrlike(self, arr):
        if not isinstance(arr, (np.ndarray, np.generic)):
            raise TypeError('dtype:{}, must be numpy array-like.'.format(type(arr)))

    @property
    def embeddings(self):
        if self._array_name_map[fields.embeddings] is None:
            print('WARNING: Get the empty embeddings array')
            return self._array_name_map[fields.embeddings]
        if len(self._array_name_map[fields.embeddings].shape) >= 3:
            print('NOTICE: Shape of given embeddings are {}, squeezed automatically.'.format(
                self._array_name_map[fields.embeddings].shape))
            self._array_name_map[fields.embeddings] = np.squeeze(self._array_name_map[fields.embeddings])
        return self._array_name_map[fields.embeddings]

    @embeddings.setter
    def embeddings(self, _embeddings):
        self._check_numpy_arrlike(_embeddings)
        if len(_embeddings.shape) >= 3:
            print('NOTICE: Shape of given embeddings are {}, squeezed automatically.'.format(
                _embeddings.shape))
            _embeddings = np.squeeze(_embeddings)
        self._array_name_map[fields.embeddings] = _embeddings

    @property
    def probabilities(self):
        if self._array_name_map[fields.probabilities] is None:
            print('WARNING: Get the empty probabilities array')
            return self._array_name_map[fields.probabilities]
        if len(self._array_name_map[fields.probabilities].shape) >= 3:
            print('NOTICE: Shape of given probabilities are {}, squeezed automatically.'.format(
                self._array_name_map[fields.probabilities].shape))
            _probabilities = np.squeeze(self._array_name_map[fields.probabilities])
        return self._array_name_map[fields.probabilities]

    @probabilities.setter
    def probabilities(self, _probabilities):
        self._check_numpy_arrlike(_probabilities)
        if len(_probabilities.shape) >= 3:
            print('NOTICE: Shape of given probabilities are {}, squeezed automatically.'.format(
                _probabilities.shape))
            _probabilities = np.squeeze(_probabilities)
        self._array_name_map[fields.probabilities] = _probabilities

    @property
    def label_ids(self):
        if self._array_name_map[fields.label_ids] is None:
            print('WARNING: Get the empty label_ids array')
        return self._array_name_map[fields.label_ids]

    @label_ids.setter
    def label_ids(self, _label_ids):
        self._check_numpy_arrlike(_label_ids)
        _label_ids = np.squeeze(_label_ids)
        self._array_name_map[fields.label_ids] = _label_ids

    @property
    def label_names(self):
        if self._array_name_map[fields.label_names] is None:
            print('WARNING: Get the empty label_names array')
        return self._array_name_map[fields.label_names]

    @label_names.setter
    def label_names(self, _label_names):
        self._check_numpy_arrlike(_label_names)
        _label_names = np.squeeze(_label_names)
        self._array_name_map[fields.label_names] = _label_names

    @property
    def instance_ids(self):
        if self._array_name_map[fields.instance_ids] is None:
            print('WARNING: Get the empty instance ids')
        return self._array_name_map[fields.instance_ids]

    @instance_ids.setter
    def instance_ids(self, _instance_ids):
        self._check_numpy_arrlike(_instance_ids)
        _instance_ids = np.squeeze(_instance_ids)
        self._array_name_map[fields.instance_ids] = _instance_ids

    @property
    def filename_strings(self):
        if self._array_name_map[fields.filename_strings] is None:
            print('WARNING: Get the empty filename strings')
        return self._array_name_map[fields.filename_strings]

    @filename_strings.setter
    def filename_strings(self, _filename_strings):
        self._check_numpy_arrlike(_filename_strings)
        self._array_name_map[fields.filename_strings] = _filename_strings


class FeatureObject(FeatureObjectBase):

    def __init__(self):
        super(FeatureObject, self).__init__()

    def load(self, data_dir):
        _npy_in_data_dir = [each for each in os.listdir(data_dir) if each.endswith('.npy')]
        for _npy in _npy_in_data_dir:
            _npy_name = re.sub('.npy', '', _npy)
            if _npy_name in self._array_name_map:
                _npy_path = '/'.join([data_dir, _npy])
                _npy_arr = np.load(_npy_path)
                self._array_name_map[_npy_name] = _npy_arr
                print('{} is loaded'.format(_npy_path))

    def save(self, data_dir):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        else:
            print('WARNING: {} is already exists, still export numpy arrays to it.'.format(data_dir))
        for _name, _arr in self._array_name_map.items():
            if _arr is not None:
                dst_path = '/'.join([data_dir, _name])
                np.save(dst_path, _arr)
