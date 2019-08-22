import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

from os import listdir
from os.path import join, isfile

import cv2
import numpy as np

import yaml
from tqdm import tqdm

# DatasetBackbone dependent
import scutils
from scutils.scdata import DatasetBackbone

from metric_learning_evaluator.inference.utils.image_utils import read_jpeg_image
from metric_learning_evaluator.inference.utils.image_utils import bbox_ratio_to_xywh
#from metric_learning_evaluator.data_tools.image_object import ImageObjectStandardFields as img_fields

from metric_learning_evaluator.inference.components.detector import Detector
from metric_learning_evaluator.core.standard_fields import DetectorStandardFields

detector_fields = DetectorStandardFields


def detection_application(configs, args):

    data_dir = args.data_dir
    out_dir = args.out_dir
    data_type = args.data_type
    detector_settings = configs['detector_settings']

    detector_num_classes = detector_settings['num_classes']
    detector_model_path = detector_settings['model_path']
    detector_labelmap_path = detector_settings['labelmap_path']
    # detector_batch_size = detector_settings['batch_size']

    detector = Detector(
        pb_model_path=detector_model_path,
        labelmap_path=detector_labelmap_path,
        num_classes=detector_num_classes)

    if os.path.exists(join(data_dir, 'dataset_backbone.db')):
        # If the input is datasetbackbone
        src_db = DatasetBackbone(data_dir)
        filenames = src_db.query_all_img_filenames()
        input_filenames = src_db.query_img_abspath_by_filename(filenames)
        print('Load input data from datasetbackbone with {} images.'.format(len(input_filenames)))
    else:
        input_filenames = [
            join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f))]
        print('Load input data from folder with {} images.'.format(len(input_filenames)))

    if out_dir is None:
        out_dir = 'tmp/output_two_stage_inference'
        print('Notice: out_dir is not give, save to {} in default.'.format(out_dir))
    dst_db = scutils.scdata.DatasetBackbone(out_dir)

    image_buffer, result_buffer = [], []
    for fn in tqdm(input_filenames):
        try:
            image = read_jpeg_image(fn)
        except:
            print('{} can not be opened properly, skip.'.format(fn))
            continue

        img_height, img_width, _ = image.shape
        detection_result = detector.detect(image)
        origin_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        det_bboxes = detection_result[detector_fields.detection_boxes]
        det_scores = detection_result[detector_fields.detection_scores]
        formal_bboxes = [bbox_ratio_to_xywh(bbox, img_height, img_width) for bbox in det_bboxes]

        img_id = dst_db.imwrite(origin_img)
        for bbox in formal_bboxes:
            anno_id = dst_db.init_annotation(img_id)
            dst_db.update_category(anno_id, 'unknown_instance')
            dst_db.update_bbox(anno_id, bbox)
    dst_db.commit()
