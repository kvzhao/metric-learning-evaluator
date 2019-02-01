"""
    Score function returns scalar values between [0, 1].
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))  # noqa

import numpy as np
from sklearn.metrics import accuracy_score


def top_k_accuracy_1d():
    pass

# TODO @kv: Is this accuracy specialized for facenet evaluation
def calculate_positive_by_distance(distances, threshold, actual_issame):
    """
      Args:
        distances, 1D numpy array
        threshold, float
        actual_issame, 1D numpy array of boolean

      Returns:
        tpr, float: True positive ratio
        fpr, float: False positive ratio
        acc, float: Accuracy = ratio of True samples over all samples
    """

    predict_issame = np.less(distances, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn) / distances.size
    return tpr, fpr, acc

def top_k_accuracy(predicted_scores, groundtruth_labels, top_k, reverse=False):
    """Calculate top-k accuracy.
    TODO: Change the name

      Args:
        scores, 2D numpy array: 
            shape = [batch_size, predicted_size]
            e.g. [
                  [],
                  [],
                  [],
                 ]

        groundtruth_labels, 2D numpy array:
            shape = [batch_size, 1]
            e.g. [[2], [3], [1], [6], ...]

        top_k, integer:

        reverse, boolean:
            TODO @kv: Set True for increasing order (order from small to large score).

      Return:
        accuracy, float:
            ratio successful predicted under given k.

    """

    accuracy = 0.0
    # TODO @kv: merge these two cases
    if top_k == 1:
        predicted_labels = np.argmax(predicted_scores, axis=1)
        accuracy = accuracy_score(groundtruth_labels, predicted_labels)
    else:
        best_k = np.argsort(predicted_scores, axis=1)[:, -top_k:]
        successes = 0
        for i in range(groundtruth_labels.shape[0]):
            if groundtruth_labels[i] in best_k[i, :]:
                successes += 1
        accuracy = float(successes)/groundtruth_labels.shape[0]
    return accuracy


def kfold_accuracy(labels, embeddings, actual_issame,
                    thresholds_in=[0,4], nrof_folds=2,):
    """Use Kfold to calculate accuracy. (Irene)
    
    Args:
        labels: [batch_size] int64 tensor of 1-indexed class.
        embeddings: [batch_size, embedding_size] float32 tensor of embedding.
        actual_issame: [batch_size/2] bool tensor whether the embedding1 and embedding2 is same.
        thresholds: the threshold of the distance between two embeddings.
        nrof_folds: number of Kfold.
    Return:
        A OrderedDict of metric  

    """

    pass


def average_precision_score(r_ls, p_ls, recall):
    """
      Args:
        r_ls, list: List of recalls
        p_ls, list: List of precisions
        recall, float: Threshold
      Return:
    """
    max_pre = .0
    if r_ls != []:
        recalls = np.array(r_ls)
        precisions = np.array(p_ls)
        for r_val in range(int(recall / .1) + 1):
            r_thr = 0.1 * r_val
            max_pre += np.max(precisions[np.where(recalls >= r_thr)])
        ap = max_pre / (int(recall / .1) + 1)
    else:
        ap = .0
    return ap
