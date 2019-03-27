import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class EvaluationStandardFields(object):
    """
      Standard declaration of evaluation objects
    """

    # ===== Evaluations =====

    # Retrival
    ranking = 'RankingEvaluation'

    # Ranking with Attributes Evaluation
    ranking_with_attrs = 'RankingWithAttributesEvaluation'

    mock = 'MockEvaluation'

    # FaceNet: Pair-wise evaluation
    facenet = 'FacenetEvaluation'

    # Avaliable evaluation implementations
    classification = 'ClassificationEvaluation'


    # ===== Inner items =====
    sampling = 'sampling'
    metric = 'metric'
    distance_measure = 'distance_measure'
    attribute = 'attribute'
    option = 'option'

    # ===== Distance measure inner items =====
    function = 'function'
    threshold = 'threshold'
    start = 'start'
    end = 'end'
    step = 'step'
