
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class AttributeStandardFields(object):
    """Collection of Known Attributes.
    """
    # Special keywords:
    attr_key = 'attribute'
    attribute = 'attribute'
    all_attributes = 'all_attributes'
    all_classes = 'all_classes'
    # none attributes are given, use class instead.
    none = 'None' 

    # General attribute types:
    category = 'category'
    supercategory = 'supercategory'
    color = 'color'
    shape = 'shape'