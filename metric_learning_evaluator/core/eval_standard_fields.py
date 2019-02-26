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
    # === Metric Function and Items ===

    mAP = 'mAP'

    # Area Under Curve
    auc = 'AUC'
    area_under_curve = 'area_under_curve'

    # Equal Error Rate
    eer = 'EER'
    equal_error_rate = 'equal_error_rate'

    # False Accepted Rate: FAR = TA(d)/P_same
    far = 'FAR'
    false_accept_rate = 'false_accept_rate'

    # Validation Rate: VAR = FA(d)/P_diff
    var = 'VAR'
    validation_rate = 'validation_rate'

    # top 1 accuracy
    accuracy = 'accuracy'
    # Top k accuracy
    top_k = 'top_k'
    
    pair_sampling = 'pair_sampling'
    ranking = 'ranking'

    distance_function = 'distance_function'
    distance_threshold = 'distance_threshold'
    path_pairlist = 'path_pairlist'
    sample_method = 'sample_method'
    sample_ratio = 'sample_ratio'
    class_sample_method = 'class_sample_method'
    ratio_of_class = 'ratio_of_class'
    ratio_of_image_per_class = 'ratio_of_image_per_class'

    # === Parameters & Internal Keys ===
    true_positive = 'true_positive'
    false_positive = 'false_positive'
    true_negative = 'true_negative'
    false_negative = 'false_negative'
    true_positive_rate = 'true_positive_rate'
    false_positive_rate = 'false_positive_rate'


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
    metric = 'metric'
    attribute = 'attribute'
    option = 'option'

    # System-related configs
    container_size = 'container_size'
    embedding_size = 'embedding_size'
    logit_size = 'logit_size'


class InputDataStandardFields(object):
    pass