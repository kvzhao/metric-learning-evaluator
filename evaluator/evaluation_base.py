"""
    The base object and data container for the metric learning evaluator.

    Brief intro:

        EmbeddingContainer: 
            Efficient object which handles the shared (globally) embedding vectors.

        AttributeContainer:
            Data object for maintaining attribute table locally in each EvaluationObejct.

        MetricEvaluationBase:

    @bird, dennis, kv
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from abc import ABCMeta
from abc import abstractmethod
import collections
from collections import namedtuple
from collections import OrderedDict
from collections import defaultdict

import logging
import numpy as np

class EmbeddingContainer(object):
    """The container for Embeddings (image_id, label_id, embedding vector)

      operations:
        - add: put one datum in the container
        - embeddings: get all embedding vectors exist in the container
        - get_embedding_by_image_ids: query embeddings by image_ids
        - get_label_by_image_ids: query labels by image_ids
        - clear: clear the internal buffer
    
      NOTE: Can we predefine the size of container
      NOTE: Implement the container with numpy for performance?

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

        # NOTE: same image_id maps to many embedding!?
        self._index_by_image_id[image_id] = self._current
        self._label_by_image_id[image_id] = label_id
        self._image_ids.append(image_id)

        self._current += 1

    def get_embedding_by_image_ids(self, image_ids):
        """Fetch batch of embedding vectors by given image ids."""
        if not (type(image_ids) is int or type(image_ids) is list):
            raise ValueError('image_ids should be int or list.')
        indices = [self._index_by_image_id[img_id] for img_id in image_ids]
        return self._embeddings[indices, ...]

    def get_label_by_image_ids(self, image_ids):
        """Fetch the labels from given image ids."""
        if type(image_ids) is int:
            return self._label_by_image_id[image_ids]
        elif type(image_ids) is list:
            return [self._label_by_image_id[img_id] for img_id in image_ids]
        else:
            raise ValueError('image_ids should be int or list.')

    @property
    def embeddings(self):
        # get embeddings up to current index
        return self._embeddings[:self._current]

    @property
    def image_ids(self):
        # get all image_ids in conatiner
        return self._image_ids

    @property
    def counts(self):
        return self._current + 1

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
    """The container for Attribute (Grouping table).

    Group: {
        "Cup": [<id1>, <id2>, <id3>],
        "Color.Red": [<id1>, <id2>, <id3>],
    }
    """

    def __init__(self):

        # attribute-id mapping, shallow key-value pair
        self._groups = defaultdict(list)

    # TODO @kv: modify to image_id: attribute_names
    def add(self, attribute_name, image_ids):
        """Add the attribute with corresponding groupping indices.

          Args:
            attribute_name, str:
                The whole query command, e.g. Color.Red or Human.Height
            image_ids, int or list.

          Logic:
            if attr not exist:
                create new key and assign the given list.
            else:
                
        """
        # type assertion
        if not(type(image_ids) is int or type(image_ids) is list):
            raise ValueError('image_ids should be int or list.')
        if type(image_ids) is int:
            image_ids = [image_ids]

        if attribute_name in self._groups:
            self._groups[attribute_name].extend(image_ids)
        else:
            self._groups[attribute_name] = image_ids


    @property
    def groups(self):
        # return attribute to image_ids
        return self._groups

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
        

class MetricEvaluationBase(object):    
    """Interface for EvaluationObject which serves as the functional module.

        Usage:

    """
    __metaclass__ = ABCMeta

    def __init__(self, per_eval_config, embedding_container, attribute_container=None):
        """Base Object for Evaluation.
          <Customized>Evaluation is the functional object which 
                      executes computation with metric functions.

          Args:
            per_eval_config, list:
                Configuration used for the EvaluationObject.
                TODO @kv: Define the format:

            embedding_container, EmbeddingContainer:
            attribute_container, AttributeContrainer:
                The attribute container can be `None` for some evaluation.

          NOTE:
            User would allocate `AttributeContainer` 
            in thier customized evaluation.
        """

        if per_eval_config and not isinstance(per_eval_config, list):
            raise ValueError('Evaluation Config is a list of required attributes.')

        # check the instance type.
        if not isinstance(embedding_container, EmbeddingContainer):
            raise ValueError('Embedded Conatiner is Needed.')
        if attribute_container and not isinstance(embedding_container, EmbeddingContainer):
            raise ValueError('Attribute Conatiner is Needed.')

        self._per_eval_config = per_eval_config
        self._embedding_container = embedding_container
        self._attribute_container = attribute_container

        # TODO: Iterator for getting embeddings from given attribute_names

    @abstractmethod
    def compute(self):
        """Compute metrics."""
        pass
