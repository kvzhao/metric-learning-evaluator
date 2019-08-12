"""ImageObject
    The image contains single or multiple instances.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from metric_learning_evaluator.core.standard_fields import ImageObjectStandardFields as img_fields


class ImageObject(object):
    """
      Object describing the image and instance:

        ImageID:0
        ImageBuffer
            - InstanceID:0
                - Box
                - Conf
                - Feature
                - Label Name
                - Label ID
            - InstanceID:1
                - Box
                - Conf
                - Feature
                - Label Name
                - Label ID
    """
    def __init__(self, image_id, raw_image):
        self.clear()

        self._image_id = image_id
        self._raw_image = raw_image

    def __repr__(self):
        _content = 'image_id:{} contains {} instances.'.format(self._image_id, len(self._instance_ids))
        for _inst_id, _inst in self._instances.items():
            _inst_bbox = _inst[img_fields.bounding_box]
            _content += '\n\t inst_id:{} @{}'.format(_inst_id, _inst_bbox)
            if img_fields.bounding_box_confidence in _inst:
                _content += '#conf={0:.3g}'.format(_inst[img_fields.bounding_box_confidence])
            if img_fields.instance_label_name in _inst:
                _content += '={}'.format(_inst[img_fields.instance_label_name])
        return _content

    def add_instance(self,
                     instance_id,
                     bbox,
                     bbox_conf=None,
                     feature=None,
                     instance_label_name=None,
                     instance_label_id=None,):
        """
        Args:
            instance_id
            bbox
            bbox_conf
            feature
        """
        if instance_id in self._instances:
            print('WARNING: INSTACE ID: {} already exist.'.format(instance_id))
        else:
            self._instance_ids.append(instance_id)
            self._instances[instance_id] = {}

        # TODO @kv: check bbox is legal
        self._instances[instance_id][img_fields.bounding_box] = bbox
        self.update_instance(instance_id, bbox_conf, feature, instance_label_name, instance_label_id)

    def update_instance(self,
                        instance_id,
                        bbox_conf=None,
                        feature=None,
                        instance_label_name=None,
                        instance_label_id=None,):
        if bbox_conf is not None:
            self._instances[instance_id][img_fields.bounding_box_confidence] = bbox_conf
        if feature is not None:
            self._instances[instance_id][img_fields.instance_feature] = feature
        if instance_label_id is not None:
            self._instances[instance_id][img_fields.instance_label_id] = instance_label_id
        if instance_label_name is not None:
            self._instances[instance_id][img_fields.instance_label_name] = instance_label_name

    @property
    def raw_image(self):
        return self._raw_image

    @property
    def image_id(self):
        return self._image_id

    @property
    def instances(self):
        return self._instances

    @property
    def instance_array(self):
        # get an 4d tensor of instances
        pass

    @property
    def instance_ids(self):
        return self._instance_ids

    @property
    def instance_bboxes(self):
        bboxes = []
        for _inst_id in self._instance_ids:
            bboxes.append(self._instances[_inst_id][img_fields.bounding_box])
        return bboxes

    @property
    def instance_label_names(self):
        return [self._instances[_inst_id][img_fields.instance_label_name]
                for _inst_id in self._instance_ids]

    def clear(self):
        self._instance_ids = []
        self._instances = {}
