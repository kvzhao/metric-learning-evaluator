"""
    The standard field defines the key elements in the returned metrics.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

class EvaluationStandardFields(object):
    """
      Standard declaration of evaluation objects
    """

    # Avaliable evaluation implementations
    classification = 'ClassificationEvaluation'

    # Retrival
    ranking = 'RankingEvaluation'

    mock = 'MockEvaluation'

    # FaceNet
    facenet = 'FacenetEvaluation'

class MetricStandardFields(object):
    """
      Standard declaration of metric function, only functions defined here can be executed.
    """
    # === Metric Functions ===

    # Top k accuracy
    top_k = 'Top_k'

    mAP = 'mAP'

    # Area Under Curve
    auc = 'AUC'

    # Equal Error Rate
    eer = 'EER'

    # False Accepted Rate
    far = 'FAR'

    # === Parameters & Internal Keys ===
    true_positive = 'true_positive'
    false_positive = 'false_positive'
    true_negative = 'true_negative'
    false_negative = 'false_negative'
    distance = 'distance'
    threshold = 'threshold'

class AttributeStandardFields(object):
    """Collection of Known Attributes.
    """
    # Special keywords:
    attr_key = 'Attr'
    all_attributes = 'All_Attributes'
    all_classes = 'All_Classes'
    # none attributes are given, use class instead.
    none = 'None' 

    # General attribute types:
    supercategory = 'supercategory'
    color = 'color'
    shape = 'shape'

class ConfigStandardFields(object):
    """Items for Evaluation Configuration
    """
    # Key of database
    database = 'database'
    # Types of database
    datasetbackbone = 'DatasetBackbone'
    zeus = 'Zeus'

    # Keywords of Evaluation & Attributes
    evaluation = 'evaluation'
    attr = 'Attr'
    labelmap = 'Labelmap' # remove this

    # System-related configs
    container_size = 'container_size'
    embedding_size = 'embedding_size'
    logit_size = 'logit_size'


class InputDataStandardFields(object):
    pass