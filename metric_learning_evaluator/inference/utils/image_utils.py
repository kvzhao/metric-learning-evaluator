"""
  Image and Bounding Box Tools
"""

import os
from os import listdir
from os.path import join
from os.path import isfile

import cv2
import numpy as np


def load_file_from_structure_folder(folder_path, ext=('.png', '.jpg', '.jpeg')):
    """Load files with extentions from structure folder
      Args:
        folder_path: string
      Return:
        filenames: List of files satisfied given image extention
        foldernames: Folder names for each files
    """
    filenames, foldernames = [], []
    for root, _, _ in os.walk(folder_path):
        img_files = load_file_from_folder(root, ext)
        filenames.extend(img_files)
        foldernames.extend([root.split('/')[-1]] * len(img_files))
    print('{} contains {} images in total'.format(folder_path, len(filenames)))
    return filenames, foldernames


def load_file_from_folder(folder_path, ext=('.png', '.jpg', '.jpeg')):
    """
      Args:
        folder_path: string
      Return:
        filenames: list of file paths with extension
    """
    filenames = [join(folder_path, f) for f in listdir(folder_path)
                 if isfile(join(folder_path, f)) and f.lower().endswith(ext)]
    print('Load {} images from {}'.format(len(filenames), folder_path))
    return filenames


def read_jpeg_image(filename, size=None):
    """
      Args:
        filename: A string of single image path
    """
    if not isinstance(filename, str):
        raise ValueError('_read_jpeg: image filename must be string, but get {}'.format(
            type(filename)))
    bgr_img = cv2.imread(filename)
    return _preprocess_image(bgr_img, size)


def _decode_jpeg(jpg_string, size):
    if not isinstance(jpg_string, str):
        raise ValueError('_decode_jpeg: jpg string must be string, but get {}'.format(
            type(jpg_string)))
    bgr_img = cv2.imdecode(jpg_string, cv2.IMREAD_COLOR)
    return _preprocess_image(bgr_img, size)


def _preprocess_image(bgr_img, size=None):
    """
    Convert to RGB image and resize image
    Args:
        bgr_img: a ndarray, BGR image
        size: a tuple, of format (height, width)
    Return:
        a RGB resized cv2 image ndarray
    """
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    shape = rgb_img.shape

    if size is not None and (shape[0] != size or shape[1] != size):
        rgb_img = cv2.resize(rgb_img, (size, size))

    return rgb_img


def convert_bbox_format(bboxes):
    """Convert box format from [x, y, w, h] to [ymin, xmin, ymax, xmax]."""
    return [[box[1], box[0], box[1]+box[3], box[0]+box[2]] for box in bboxes]


def crop_and_resize(image, bbox, size=224):
    """Read bounding boxes and return the cropped image.
        bbox: [xmin, ymin, box_w, box_h]
    """
    img = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    return cv2.resize(img, (size, size))


def crop(image, bbox):
    return image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]


def bbox_ratio_to_xywh(raw_bbox, img_height, img_width):
    # Scale bbox with respect to each images' shape
    ymin, xmin, ymax, xmax = raw_bbox
    xmin = int(xmin * img_width)
    ymin = int(ymin * img_height)
    xmax = int(xmax * img_width)
    ymax = int(ymax * img_height)
    return [xmin, ymin, abs(xmax - xmin), abs(ymax - ymin)]


def bbox_xywh_to_corner_format(bbox):
    """Convert box format from [x, y, w, h] to [ymin, xmin, ymax, xmax]."""
    return [bbox[1], bbox[0], bbox[1]+bbox[3], bbox[0]+bbox[2]]


def bboxes_xywh_to_corner_format(bboxes):
    """Convert box format from [x, y, w, h] to [ymin, xmin, ymax, xmax]."""
    return [[box[1], box[0], box[1]+box[3], box[0]+box[2]] for box in bboxes]


def bbox_size_offset(bbox, offset):
    """
      Args:
        bbox: [xmin, ymin, box_w, box_h]
        offset: int pixel
    """
    return [bbox[0], bbox[1], bbox[2]-offset, bbox[3]-offset]


def bbox_center_offset(bbox, offset_x=0, offset_y=0):
    """
      Args:
        bbox: [xmin, ymin, box_w, box_h]
        offset: int pixel
    """
    return [bbox[0]+offset_x, bbox[1]+offset_y, bbox[2], bbox[3]]