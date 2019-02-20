"""
    Define data containers for the metric learning evaluator.

    Brief intro:

        EmbeddingContainer: 
            Efficient object which handles the shared (globally) embedding vectors.

        AttributeContainer:
            Data object for maintaining attribute table in each EvaluationObejct.

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

from core.eval_standard_fields import ConfigStandardFields as config_fields

class EmbeddingContainer(object):
    """The Data Container for Embeddings & Logit (image_id, label_id, embedding vector).

      operations:
        - add: put one datum in the container
        - embeddings: get all embedding vectors exist in the container
        - get_embedding_by_image_ids: query embeddings by image_ids
        - get_label_by_image_ids: query labels by image_ids
        - clear: clear the internal buffer
    
      NOTE: We CAN NOT confirm the orderness of logits & embedding consistent with image_ids.
      TODO @kv: implement save & load.
      TODO @kv: Error-hanlding when current exceeds container_size

    """
    def __init__(self, embedding_size, logit_size, container_size=10000):
        """Constructor of the Container.

          Args:
            embedding_size, int:
                Dimension of the embedding vector, e.g. 1024 or 2048.
            logit_size, int:
                Disable this by giving size equals to 0.
            container_size, int:
                Number of embedding vector that container can store.
        
        """
        self._embedding_size = embedding_size
        self._logit_size = logit_size
        self._container_size = container_size
        # logits, prelogits (embeddeing),
        self._embeddings = np.empty((container_size, embedding_size), dtype=np.float32)
        if logit_size == 0:
            self._logits = None
        else:
            self._logits = np.empty((container_size, logit_size), dtype=np.float32)
        self._label_by_image_id = {}
        self._index_by_image_id = {}
        self._image_id_by_label = defaultdict(list)
        # orderness is maintained in _image_ids
        self._image_ids = []

        self._current = 0
    
    def add(self, image_id, label_id, embedding, logit=None):
        """Add image_id, label_id and embeddings.
        TODO: Add one more argument: logit
          Args:
            image_id, int:
                Unique image_id which can not be repeated in the container.
            label_id, int:
                Index of given class corresponds to the image.
            embedding, numpy array:
                One dimensional embedding vector with size less than self._embedding_size.
            (optional) logit, numpy array:
                One dimensional vector.
        """

        # assertions: embedding size, 
        assert embedding.shape[0] <= self._embedding_size, "Size of embedding vector is greater than the default."
        # TODO @kv: Also check the logit size, and if it exists.

        # NOTE @kv: Do we have a better round-off?
        assert self._current < self._container_size, "The embedding container is out of capacity!"

        if not isinstance(embedding, (np.ndarray, np.generic)):
            raise ValueError('Legal dtype of embedding is numpy array.')

        self._embeddings[self._current, ...] = embedding

        if not logit is None:
            self._logits[self._current, ...] = logit

        # NOTE: same image_id maps to many embedding!?
        self._index_by_image_id[image_id] = self._current
        self._label_by_image_id[image_id] = label_id
        self._image_id_by_label[label_id].append(image_id)
        self._image_ids.append(image_id)

        self._current += 1

    def get_embedding_by_image_ids(self, image_ids):
        """Fetch batch of embedding vectors by given image ids."""
        if not (type(image_ids) is int or type(image_ids) is list):
            raise ValueError('image_ids should be int or list.')
        if isinstance(image_ids, int):
            image_ids = [image_ids]
        indices = [self._index_by_image_id[img_id] for img_id in image_ids]
        return self._embeddings[indices, ...]

    def get_label_by_image_ids(self, image_ids):
        """Fetch the labels from given image_ids."""
        if isinstance(image_ids, list):
            return [self._label_by_image_id[img_id] for img_id in image_ids]
        else:
            return self._label_by_image_id[image_ids]
            #raise ValueError('image_ids should be int or list.')

    def get_image_ids_by_label(self, label_id):
        """Fetch the image_ids from given label_id."""
        if not np.issubdtype(type(label_id), np.integer):
            raise ValueError('Query label id should be integer.')
        return self._image_id_by_label[label_id]

    @property
    def embeddings(self):
        # get embeddings up to current index
        return self._embeddings[:self._current]

    @property
    def logits(self):
        # get logits up to current index
        return self._logits[:self._current]

    @property
    def image_ids(self):
        # get all image_ids in container
        return self._image_ids

    @property
    def image_id_groups(self):
        return self._image_id_by_label

    @property
    def index_by_image_ids(self):
        return self._index_by_image_id

    @property
    def counts(self):
        return self._current

    def clear(self):
        # clear dictionaries
        self._index_by_image_id = {}
        self._label_by_image_id = {}
        self._image_ids = []
        
        # reset the current index, rewrite array instead of clear elements.
        self._current = 0

    def save(self):
        pass

    def load(self):
        pass

class AttributeContainer(object):
    """The Data Container for Attributes (Grouping table).

      Usage:
        User would add image_id & corresponding attributes into container
        then, the managed structure can be returned.

      case 1:
        Groups: {
            "Shape.Cup": [<image_id:1>, <image_id:2>, <image_id:3>],
            "Color.Red": [<image_id:1>, <image_id:3>, <image_id:5>, ...],
        }

      case 2:
        ImageAttributes: {
            image_id: ["Shape.Bottle", "Color.Blue", ...],
        }

      Operations:
        - add: put one datum into container

    """

    def __init__(self):
        """
          Internal data structure:
            groups:
                The map of attribute_name to image_ids.
                e.g. group[attribute_name] = [list of image_ids] (whose attribute is the same.)
            image2attr:
                The map of image_id to attributes.
                e.g. image2attr[image_id] = [list of attributes in string].
        """

        # attribute-id mapping, shallow key-value pair
        self._groups = defaultdict(list)
        self._image2attr = defaultdict(list)

    # TODO @kv: modify to image_id: attribute_names
    def add(self, image_id, attributes):
        """Add the attribute with corresponding groupping indices.

          Args:
            image_id, int:
                An image_id is used for query attributes.

            attributes, list of str:
                Attributes of the given image_id.

          Logic:
            if attr not exist:
                create new key and assign the given list.
            else:
                
        """
        # type assertion
        if not type(image_id) is int:
            raise ValueError('image_id should be an integer.')
        if isinstance(attributes, str):
            attributes = [attributes]
        
        if not all(isinstance(_attr, str) for _attr in attributes):
            raise ValueError('attributes type should be str or list of str.')
        
        self._image2attr[image_id] = attributes

        for _attr in attributes:
            if _attr in self._groups:
                self._groups[_attr].append(image_id)
            else:
                self._groups[_attr] = [image_id]


    @property
    def groups(self):
        """Return attribute to image_ids
            e.g. {'Color.Red': [0, 2, 5, ...], }
        """
        return self._groups

    @property
    def image_attributes(self):
        """Return image_id to attributes
            e.g. {'2': ['Color.Red', 'Shape.Bottle', ...], }
        """
        return self._image2attr

    @property
    def attr_lookup(self):
        # return image_id to attributes
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
        self._image2attr = defaultdict(list)

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
                Generated from ConfigParser.get_per_eval_metrics()

            attributes, list of str:
                Generated from ConfigParser.get_per_eval_attributes()

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
        return self._results

    def clear(self):
        self._results = {}