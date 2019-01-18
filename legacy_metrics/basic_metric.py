"""
    Functions here would be legacy.
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.metrics import metric_variable
from tensorflow.python.ops import state_ops
from nets import facenet
from collections import OrderedDict

def split_embedding(embedds):
    embeddings1 = embedds[0::2]
    embeddings2 = embedds[1::2]
    batch_size = embeddings1.shape[0]
    return embeddings1, embeddings2, actual_issame

def basic_metric(labels, embeddings, actual_issame,
                 thresholds=[0,4], nrof_folds=2,):
    """Use Kfold to calculate accuracy
    
    Args:
        labels: [batch_size] int64 tensor of 1-indexed class.
        embeddings: [batch_size, embedding_size] float32 tensor of embedding.
        actual_issame: [batch_size/2] bool tensor whether the embedding1 and embedding2 is same.
        thresholds: the threshold of the distance between two embeddings.
        nrof_folds: number of Kfold.
    Return:
        A OrderedDict of metric  

    """
    # split embeddings using py_func
    embedding1, embedding2 = tf.py_func(split_embedding,
                                        [embeddings], 
                                        [tf.float32, tf.float32])
    
    # transfer thresholds to tensor
    thresholds_np = np.arange(thresholds[0], thresholds[1], 0.01)
    thresholds = tf.convert_to_tensor(thresholds_np, dtype=tf.float32)
    
    # transfer number of Kfold to tensor
    nrof_folds_np = np.array([nrof_folds])
    nrof_folds = tf.convert_to_tensor(nrof_folds_np, dtype=tf.int32)
    folds_len = nrof_folds_np[0]
    
    # calculate accuracy using py_func
    tprs, fprs, accuracys = tf.py_func(facenet.calculate_roc,
                                       [thresholds, embedding1, embedding2, actual_issame, nrof_folds],
                                       [tf.float32, tf.float32, tf.float64])
    # define metric_ops as OrderedDict
    metric_ops = OrderedDict()
    
    # split the accuracy to a list of tensor
    acc_split = tf.unstack(accuracys, num=folds_len, axis=0)

    # geneerate the metric_ops
    for idx in range(folds_len):
        # calculate the mean accuracy and generate op, then add the them to metric_op
        m_accuracy, update_accuracy_op = tf.metrics.mean(acc_split[idx])
        metric_ops['{}-fold'.format(idx)] = (m_accuracy, update_accuracy_op)
    
    return metric_ops
   
    






