import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

class MetricStandardFields(object):
    """
      Standard declaration of metric function, only functions defined here can be executed.
    """
    # === Metric Function and Items ===
    metric = 'metric'

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

    accuracy = 'accuracy'
    # Top k accuracy
    top_k_accuracy = 'top_k_accuracy'
    top_k_hit_accuracy = 'top_k_hit_accuracy'
    
    pair_sampling = 'pair_sampling'
    ranking = 'ranking'

    # deprecated
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

    # === Distance Measures ===
