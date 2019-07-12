"""
  The script renders top_k ranking results and save them into given folder.
  Also, this script is defined as an analysis tool

  Input:
    DatasetBackbone with extracted embeddings
  Output:
    Folder with images
"""

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))  # noqa

import matplotlib.pyplot as plt
"""
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
from matplotlib.font_manager import _rebuild
_rebuild()
"""

from scutils.scdata import DatasetBackbone

import cv2
import yaml
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from metric_learning_evaluator.config_parser.parser import ConfigParser

from metric_learning_evaluator.data_tools.feature_object import FeatureObject
from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
from metric_learning_evaluator.utils.io_utils import create_embedding_container_from_featobj

from metric_learning_evaluator.query.general_database import QueryInterface
from metric_learning_evaluator.data_tools.attribute_table import AttributeTable

from metric_learning_evaluator.index.agent import IndexAgent

from metric_learning_evaluator.metrics.ranking_metrics import RankingMetrics
from metric_learning_evaluator.utils.sample_strategy import SampleStrategy
from metric_learning_evaluator.utils.result_saver import ResultSaver

"""
  1. Load datasetbackbone; Load retrieved embeddings
  2. ? create a mapping between instance_id & filename_string
  3. perform ranking
  4. show & save if it fails
"""

def save_multiple_images(images, cols=1, titles=None, save_path=None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(u'{}'.format(title))
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print('save figure to {}'.format(save_path))
    plt.clf()
    fig = None

def main(args):
    json_file_path = args.json_file_path
    data_dir = args.data_dir
    database_dir = args.database_dir
    output_dir = args.output_dir

    if not output_dir and database_dir:
        output_dir = '/tmp/query_retrieved_samples'

    if not os.path.exists(output_dir) and database_dir:
        os.makedirs(output_dir)

    if data_dir:
        featobj = FeatureObject()
        featobj.load(data_dir)
        label_ids = featobj.label_ids
        label_names = featobj.label_names
        labelmap = {}
        for _id, _name in zip(label_ids, label_names):
            if _id not in labelmap:
                labelmap[_id] = _name
    if database_dir:
        # load dataset backbone
        src_db = DatasetBackbone(database_dir)

    saver = ResultSaver()
    rank_events = saver.load(json_file_path)

    misclassified = defaultdict(set)
    misclassified_instances = defaultdict(list)
    # need a labelmap

    for event in rank_events:
        query_label = event['query_label']
        retrieved_labels = event['retrieved_labels']
        query_instance = event['query_instance']
        retrieved_instances = event['retrieved_instances']
        retrieved_distances = event['retrieved_distances']
        top_1_retrieved = retrieved_labels[0]
        top_1_retrieved_instance = retrieved_instances[0]
        if data_dir:
            top_1_retrieved = labelmap[top_1_retrieved]
            query_label = labelmap[query_label]
            retrieved_labels = [labelmap[_label] for _label in retrieved_labels]
        misclassified[query_label].add(top_1_retrieved)
        misclassified_instances[query_instance].append(retrieved_instances)

    if database_dir:
        print('Start saving images...')
        for _query, _retrieved_list in tqdm(misclassified_instances.items()):
            _query_info = src_db.query_anno_info_by_anno_id(_query)[0]
            _query_filename = _query_info['filename']
            _query_category_name = _query_info['category']
            for _retrieved in _retrieved_list:
                if len(_retrieved) > 5:
                    continue
                image_paths, titles, instances = [], [], []
                image_paths.append(src_db.query_img_abspath_by_filename(_query_filename)[0])
                instances.append(_query)
                titles.append(_query_category_name)
                for _top_k in _retrieved:
                    _retrieved_info = src_db.query_anno_info_by_anno_id(_top_k)[0]
                    image_paths.append(src_db.query_img_abspath_by_filename(
                        _retrieved_info['filename'])[0])
                    instances.append(_top_k)
                    titles.append(_retrieved_info['category'])
                images = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in image_paths]
                title_str = '_'.join(titles)
                name = '{}_{}.jpeg'.format(title_str, _query)
                out_path = output_dir + '/' + name
                save_multiple_images(images, titles=instances, save_path=out_path)
                plt.close('all')
        print('Done.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--json_file_path', type=str, default=None)
    parser.add_argument('-dd', '--data_dir', type=str, default=None)
    parser.add_argument('-db', '--database_dir', type=str, default=None)
    parser.add_argument('-od', '--output_dir', type=str, default=None)

    args = parser.parse_args()
    main(args)