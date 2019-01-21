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

    evaluation_name = 'evaluation_name'
    metric_type = 'metric_type'

    # Returned formats, Type of metrics
    accuarcy = 'accuracy'

class EvalConfigStandardFields(object):
    database = 'database'
    evaluation = 'evaluation'

    # Types of database
    datasetbackbone = 'DatasetBackbone'
    zeus = 'Zeus'

    # Types of evaluation
    mock = 'mock'
    ranking = 'ranking'
    classification = 'classification'


    # NOTE: maybe we do not need to predefine these requests
    # Types of known query attributes
    overall = 'Overall' # no attribute is needed.
    all_attributes = 'AllAttributes'
    supercategory = 'supercategory'
    color = 'color'
    shape = 'shape'

    # System-related configs
    container_size = 'container_size'
    embedding_size = 'embedding_size'
    logit_size = 'logit_size'