import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

from os import listdir
from os.path import join, isfile

import cv2
import time
import numpy as np
from skimage.draw import polygon

import yaml
from tqdm import tqdm

# DatasetBackbone dependent
import scutils
from scutils.scdata import DatasetBackbone
from scutils.scimage import convert_contours_coco2opencv
from skimage import measure
from sklearn.utils.linear_assignment_ import linear_assignment

from metric_learning_evaluator.inference.utils.image_utils import read_jpeg_image
from metric_learning_evaluator.inference.utils.image_utils import bbox_ratio_to_xywh
#from metric_learning_evaluator.data_tools.image_object import ImageObjectStandardFields as img_fields

from metric_learning_evaluator.inference.components.detector import Detector
from metric_learning_evaluator.inference.components.detector_base import DetectorStandardFields
from metric_learning_evaluator.inference.components.mask_detector import MaskDetector
from metric_learning_evaluator.inference.components.mask_detector_base import MaskDetectorStandardFields

detector_fields = MaskDetectorStandardFields

def mapping_application(configs, args):

    # input output setting
    data_dir = args.data_dir
    out_dir = args.out_dir
    matching_iou_threshold = args.matching_iou_threshold

    if os.path.exists(join(data_dir, 'dataset_backbone.db')):
        # If the input is datasetbackbone
        src_db = DatasetBackbone(data_dir)
        filenames = src_db.query_all_img_filenames()
        #input_filenames = src_db.query_img_abspath_by_filename(filenames)
        print('Load input data from datasetbackbone with {} images.'.format(len(filenames)))
    else:
        input_filenames = [
            join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f))]
        print('Load input data from folder with {} images.'.format(len(input_filenames)))

    if out_dir is None:
        out_dir = 'tmp/output_two_stage_inference'
        print('Notice: out_dir is not give, save to {} in default.'.format(out_dir))
    dst_db = scutils.scdata.DatasetBackbone(out_dir)

    # detector setting
    detector_settings = configs['detector_settings']

    detector_num_classes = detector_settings['num_classes']
    detector_model_path = detector_settings['model_path']
    detector_labelmap_path = detector_settings['labelmap_path']
    detector_image_height = detector_settings['image_height']
    detector_image_width = detector_settings['image_width']

    detector = MaskDetector(
        pb_model_path=detector_model_path,
        labelmap_path=detector_labelmap_path,
        num_classes=detector_num_classes,
        image_height=detector_image_height,
        image_width=detector_image_width)
    
    # calculat
    for fn in tqdm(filenames):
        try:
            abs_fn = os.path.join(src_db.img_dir, fn)
            image = read_jpeg_image(abs_fn)
        except:
            print('{} can not be opened properly, skip.'.format(fn))
            continue

        img_height, img_width, _ = image.shape
        #start = time.time()
        detection_result = detector.detect(image)
        #print("detector cost: ", time.time()-start)
        origin_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        gt_results = {"binary_masks": [],
                     "bboxes": [],
                     "categories": []}
        det_results = {"binary_masks": [],
                      "bboxes": []}
        results = {"binary_masks": [],
                   "bboxes": [],
                   "categories": []}

        #start = time.time()
        gt_results["binary_masks"], gt_results["bboxes"], gt_results["categories"] = fetch_db(src_db, fn, img_height, img_width) 
        #print("fetch db cost: ", time.time()-start)
        
        #start = time.time()
        det_results["binary_masks"], det_results["bboxes"] = fetch_detection_result(detection_result, img_height, img_width)
        #print("fetch results cost: ", time.time()-start)

        #start = time.time()
        results = calculate_iou_matrix(gt_results, det_results, matching_iou_threshold)
        #print("iou cost: ", time.time()-start)

        #start = time.time()
        dump_result(origin_img, dst_db, results, detection_result)
        #print("dump cost: ", time.time()-start)
        
    dst_db.commit()
    #check_instance(src_db, dst_db)    

def fetch_detection_result(detection_result, img_height, img_width):
    masks_buffer, bboxes_buffer = [], []
    det_bboxes = detection_result[detector_fields.detection_boxes]
    formal_bboxes = [bbox_ratio_to_xywh(bbox, img_height, img_width) for bbox in det_bboxes]
    result_lenth = len(formal_bboxes)

    for idx in range(result_lenth):
        bbox = formal_bboxes[idx]
        bboxes_buffer.append(bbox)
        if detector_fields.detection_masks in detection_result.keys():
            mask = detection_result[detector_fields.detection_masks][idx]
            mask = cv2.resize(mask, (img_width, img_height))
            binary_mask = np.where(mask > 0, 1, 0)
            masks_buffer.append(binary_mask) 
        else:
            binary_mask = DBbbox2binarymask(img_width, img_height, bbox)
            masks_buffer.append(binary_mask)
    return masks_buffer, bboxes_buffer 

def fetch_db(src_db, filename, img_height, img_width):
    anno_infos = src_db.query_anno_info_by_filename(filename)
    masks_buffer, bboxes_buffer, categories_buffer = [], [], []
    for anno_info in anno_infos:
        categories_buffer.append(anno_info["category"])
        bboxes_buffer.append(anno_info["bbox"])
        if anno_info["segmentation"] is not None:
            masks_buffer.append(DBseg2binarymask(img_width, img_height, anno_info["segmentation"]))
        else:
            masks_buffer.append(DBbbox2binarymask(img_width, img_height, anno_info["bbox"])) 
    return masks_buffer, bboxes_buffer, categories_buffer
            
def calculate_iou_matrix(gt_results, det_results, iou_threshold):
    gt_binarymasks = gt_results["binary_masks"]
    gt_categories = gt_results["categories"]
    det_binarymasks = det_results["binary_masks"]
    det_bboxes = det_results["bboxes"]
    iou_matrix = np.zeros((len(gt_binarymasks), len(det_binarymasks)), dtype=np.float32)
    for gt_idx, gt_bm in enumerate(gt_binarymasks):
        for det_idx, det_bm in enumerate(det_binarymasks):
            iou_matrix[gt_idx, det_idx] = get_iou_score(gt_bm, det_bm)
    
    matched_indices = linear_assignment(-iou_matrix)

    categories_buffer = ["unknown"]*len(gt_binarymasks)
    for m in matched_indices:
        gt_idx, det_idx = m[0], m[1]
        if (iou_matrix[gt_idx, det_idx] < iou_threshold):
            pass
        else:
            categories_buffer[gt_idx] = gt_categories[gt_idx]
            gt_results["binary_masks"][gt_idx] = det_binarymasks[det_idx]
            gt_results["bboxes"][gt_idx] = det_bboxes[det_idx]
            
    results = {}
    results["binary_masks"] = gt_results["binary_masks"]
    results["bboxes"] = gt_results["bboxes"]
    results["categories"] = categories_buffer
    return results

def dump_result(origin_img, dst_db, results, detection_result):
    result_lenth = len(results["bboxes"])
    
    img_id = dst_db.imwrite(origin_img)
    for idx in range(result_lenth):
        cate = results["categories"][idx]
        bbox = results["bboxes"][idx]
        mask = results["binary_masks"][idx]

        anno_id = dst_db.init_annotation(img_id)
        dst_db.update_category(anno_id, cate)
        dst_db.update_bbox(anno_id, bbox)
        if detector_fields.detection_masks in detection_result.keys():
            poly = binarymask_to_segDBform(mask)
            dst_db.update_poly(anno_id, [poly])
    
def binarymask_to_segDBform(binary_mask):
    '''Note that skimage.measure.find_contours return a [[y,x][y,x]......]'''
    '''http://scikit-image.org/docs/0.5/auto_examples/plot_contours.html'''
    contours = measure.find_contours(binary_mask, 0.9)
    segDBform=[]
    for i in range(len(contours[0])):
        x = int(contours[0][i][1])
        y = int(contours[0][i][0])
        segDBform.extend([x])
        segDBform.extend([y])
    return segDBform
        
        
def DBbbox2binarymask(w, h, db_bbox):
    '''
    http://scikit-image.org/docs/0.7.0/auto_examples/plot_shapes.html
    '''
    binary_mask = np.zeros((h, w))

    up_left = [db_bbox[1], db_bbox[0]]
    down_left = [db_bbox[1]+db_bbox[3], db_bbox[0]]
    down_right = [db_bbox[1]+db_bbox[3], db_bbox[0]+db_bbox[2]]
    up_right = [db_bbox[1], db_bbox[0]+db_bbox[2]]

    poly = np.array([
        up_left,
        down_left,
        down_right,
        up_right
    ])
    rr, cc = polygon(poly[:,0], poly[:,1], binary_mask.shape) 
    binary_mask[rr,cc] = 1
    return binary_mask

def DBseg2binarymask(w, h, contours):
    '''
    http://scikit-image.org/docs/0.7.0/auto_examples/plot_shapes.html
    '''
    binary_mask  = np.zeros((h, w))
    contours = convert_contours_coco2opencv(contours)
    cv2.fillPoly(binary_mask, pts = contours, color = (1))
    return binary_mask

def get_iou_score(gt_binmsk, pd_binmsk):
    '''
    gt_binmsk: groundtruth binary mask, shape= (w,h) # no channel
    pd_binmsk: prediction binary mask,  shape= (w,h) # no channel
    
    Definition of IOU(intersection over union):
    IOU = intersection/union
    '''
    gt_area = np.sum(gt_binmsk == 1)
    pd_area = np.sum(pd_binmsk == 1)
    intersection = np.sum((gt_binmsk == 1) * (pd_binmsk == 1))
    union = gt_area+pd_area-intersection
    
    iou_score = intersection / union
    return iou_score

def check_instance(src_db, dst_db):
    src_annos = src_db.query_anno_id_by_filename(src_db.query_all_img_filenames())
    dst_annos = dst_db.query_anno_id_by_filename(dst_db.query_all_img_filenames())

    if len(src_annos) != len(dst_annos):
        print("number of annotation is not consistent src: {}, dst{}".format(len(src_annos), len(dst_annos)))
