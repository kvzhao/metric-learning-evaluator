"""
  Embedding Data Reader
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

class EmbeddingDataStandardFields:
    embeddings = 'embeddings'
    label_ids = 'label_ids'
    instance_ids = 'instance_ids'
    filename_strings = 'filename_strings'
    super_labels = 'super_labels'

fields = EmbeddingDataStandardFields

def get_var_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [k for k, v in callers_local_vars if v is var][0]

class EmbeddingDataBase(object):

    def __init__(self):

        self._array_names = [
            fields.embeddings,
            fields.label_ids,
            fields.instance_ids,
            fields.filename_strings,
            fields.super_labels]
        self._array_name_map = {
            fields.embeddings: None,
            fields.label_ids: None,
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
            raise TypeError ('dtype:{}, must be numpy array-like.'.format(type(arr)))

    @property
    def embeddings(self):
        if self._array_name_map[fields.embeddings] is None:
            print ('WARNING: Get the empty embeddings array')
        return self._array_name_map[fields.embeddings]

    @embeddings.setter
    def embeddings(self, _embeddings):
        self._check_numpy_arrlike(_embeddings)
        self._array_name_map[fields.embeddings] = _embeddings

    @property
    def label_ids(self):
        if self._array_name_map[fields.label_ids] is None:
            print ('WARNING: Get the empty label_ids array')
        return self._array_name_map[fields.label_ids]

    @label_ids.setter
    def label_ids(self, _label_ids):
        self._check_numpy_arrlike(_label_ids)
        self._array_name_map[fields.label_ids] = _label_ids

    @property
    def instance_ids(self):
        if self._array_name_map[fields.instance_ids] is None:
            print ('WARNING: Get the empty instance ids')
        return self._array_name_map[fields.instance_ids]

    @instance_ids.setter
    def instance_ids(self, _instance_ids):
        self._check_numpy_arrlike(_instance_ids)
        self._array_name_map[fields.instance_ids] = _instance_ids

    @property
    def filename_strings(self):
        if self._array_name_map[fields.filename_strings] is None:
            print ('WARNING: Get the empty filename strings')
        return self._array_name_map[fields.filename_strings]

    @filename_strings.setter
    def filename_strings(self, _filename_strings):
        self._check_numpy_arrlike(_filename_strings)
        self._array_name_map[fields.filename_strings] = _filename_strings

    @property
    def super_labels(self):
        return self._super_labels

    @super_labels.setter
    def super_labels(self, _super_labels):
        self._check_numpy_arrlike(_super_labels)
        self._super_labels = _super_labels

class EmbeddingDataObject(EmbeddingDataBase):

    def __init__(self):
        super(EmbeddingDataObject, self).__init__()

    def load(self, data_dir):

        _npy_in_data_dir = [each for each in os.listdir(data_dir) if each.endswith('.npy')]

        for _npy in _npy_in_data_dir:
            _npy_name = re.sub('.npy', '', _npy)
            if _npy_name in self._array_name_map:
                _npy_path = '/'.join([data_dir, _npy])
                _npy_arr = np.load(_npy_path)
                #self._array_name_map[_npy_name] = np.load(_npy_path)
                self._array_name_map[_npy_name] = _npy_arr
                print ('{} is loaded'.format(_npy_path))

    def save(self, data_dir):
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        else:
            print ('WARNING: {} is already exists, still export numpy arrays to it.'.format(
                data_dir))

        for _name, _arr in self._array_name_map.items():
            if not _arr is None:
                dst_path = '/'.join([data_dir, _name])
                np.save(dst_path, _arr)