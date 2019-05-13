"""
    Define data containers for the metric learning evaluator.

    Brief intro:

        AttributeContainer:
            Data object for maintaining attribute table in each EvaluationObject.

    @bird, dennis, kv
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

from metric_learning_evaluator.query.general_database import QueryInterface

class AttributeContainer(object):
    """The Data Container for Attributes (domain & property)

      Domain:
      Property:

      Usage:
        User would add instance_id & corresponding attributes into container
        then, the managed structure can be returned.

      case 1:
        Groups: {
            "Shape.Cup": [<instance_id:1>, <instance_id:2>, <instance_id:3>],
            "Color.Red": [<instance_id:1>, <instance_id:3>, <instance_id:5>, ...],
        }

      case 2:
        ImageAttributes: {
            instance_id: ["Shape.Bottle", "Color.Blue", ...],
        }

      Operations:
        - add: put one datum into container

    """

    def __init__(self):
        """
          Internal data structure:
            groups:
                The map of attribute_name to instance_ids.
                e.g. group[attribute_name] = [list of instance_ids] (whose attribute is the same.)
            instance2attr:
                The map of instance_id to attributes.
                e.g. instance2attr[instance_id] = [list of attributes in string].
        """

        # attribute-id mapping, shallow key-value pair
        self._groups = defaultdict(list)
        self._instance2attr = defaultdict(list)

    # TODO @kv: modify to instance_id: attribute_names
    def add(self, instance_id, attributes):
        """Add the attribute with corresponding groupping indices.

          Args:
            instance_id, int:
                An instance_id is used for query attributes.

            attributes, list of str:
                Attributes of the given instance_id.

          Logic:
            if attr not exist:
                create new key and assign the given list.
            else:
                
        """
        # type assertion
        if not type(instance_id) is int:
            raise ValueError('instance_id should be an integer.')
        if isinstance(attributes, str):
            attributes = [attributes]
        
        if not all(isinstance(_attr, str) for _attr in attributes):
            raise ValueError('attributes type should be str or list of str.')
        
        self._instance2attr[instance_id] = attributes

        for _attr in attributes:
            if _attr in self._groups:
                self._groups[_attr].append(instance_id)
            else:
                self._groups[_attr] = [instance_id]


    @property
    def groups(self):
        """Return attribute to instance_ids
            e.g. {'Color.Red': [0, 2, 5, ...], }
        """
        return self._groups

    @property
    def instance_attributes(self):
        """Return instance_id to attributes
            e.g. {'2': ['Color.Red', 'Shape.Bottle', ...], }
        """
        return self._instance2attr

    @property
    def attr_lookup(self):
        # return instance_id to attributes
        _attr_lookup = defaultdict(list)
        for _attr, _img_ids in self._groups.items():
            for _id in _img_ids:
                _attr_lookup[_id].append(_attr)
        return _attr_lookup

    @property
    def attr_name_lookup(self):
        """Dynamically generate `attr_name` to `index` 
            mapping for metric function.
            NOTE: It will be risky if the function is called
                  before the whole data added.
        """
        _attrs = sorted(self._groups.keys())
        _attr_name_lookup = {}
        for _id, _attr in enumerate(_attrs):
            _attr_name_lookup[_attr] = _id
        return _attr_name_lookup

    def clear(self):
        # clear the internal dict.
        self._groups = defaultdict(list)
        self._instance2attr = defaultdict(list)
        print ('Clear attribute container.')