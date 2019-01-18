"""
    The standard field defines the key elements in the returned metrics.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

class EvalStandardFields(object):
    # input data
    image_id = 'image/id'
    image_class_text = 'image/class/text'
    image_class_label = 'image/class/label'


class EvalConfigStandardFields(object):
    database = 'database'
    evaluation = 'evaluation'

    # types of database
    datasetbackbone = 'DatasetBackbone'
    zeus = 'Zeus'

    # types of evaluation
    mock = 'mock'
    ranking = 'ranking'
    classification = 'classification'

    # types of metric
    accuarcy = 'accuracy'

    # NOTE: maybe we do not need to predefine these requests
    # query requests
    supercategory = 'supercategory'
    # for example.
    color = 'color'

    # system related
    container_size = 'container_size'
    embedding_size = 'embedding_size'
    logit_size = 'logit_size'