import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.metrics import metric_variable
from tensorflow.python.ops import state_ops
from nets import facenet
from collections import OrderedDict

def split_embedding(embedds):
    embeddings1 = embedds[0::2]
    embeddings2 = embedds[1::2]
    return embeddings1, embeddings2

def irene_metric(labels, embeddings, actual_issame,
                 thresholds_in=[0,4], nrof_folds=2,):
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
    # collect embedding  
    embedding_concat, embedding_concat_op = tf.contrib.metrics.streaming_concat(embeddings)
    
    # split embeddings using py_func
    embedding1, embedding2 = tf.py_func(split_embedding,
                                        [embedding_concat], 
                                        [tf.float32, tf.float32])
    
    # transfer thresholds to tensor
    thresholds_np = np.arange(thresholds_in[0], thresholds_in[1], 0.01)
    thresholds = tf.convert_to_tensor(thresholds_np, dtype=tf.float32)
    
    # transfer number of Kfold to tensor
    nrof_folds_np = np.array([nrof_folds])
    nrof_folds = tf.convert_to_tensor(nrof_folds_np, dtype=tf.int32)
    folds_len = nrof_folds_np[0]
    
    # calculate accuracy using py_func
    tprs, fprs, accuracys = tf.py_func(facenet.calculate_roc,
                                       [thresholds, embedding1, embedding2, actual_issame, nrof_folds],
                                       [tf.float32, tf.float32, tf.float64])
    # calculate mean and std of accuracys
    mean, std = tf.nn.moments(accuracys, axes=0)

    # transfer thresholds to tensor
    thresholds_np = np.arange(thresholds_in[0], thresholds_in[1], 0.001)
    thresholds = tf.convert_to_tensor(thresholds_np, dtype=tf.float32)
    
    far_target_np = np.array([1e-3])
    far_target = tf.convert_to_tensor(far_target_np, dtype=tf.float32)

    val, val_std, far = tf.py_func(facenet.calculate_val,
                                   [thresholds, embedding1, embedding2, actual_issame, nrof_folds],
                                   [tf.float64, tf.float64, tf.float64])
    
    # define metric_ops as OrderedDict
    metric_ops = OrderedDict()
    metric_ops["mean_std"] = ([mean, std, val, val_std, far], embedding_concat_op)
    
    return metric_ops
