"""
  Dump annotations and attribute to a csv file.
"""

import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from scutils.scdata import DatasetBackbone
import pandas as pd


def main(args):
    src_db_path = args.data_dir
    labelmap_path = args.labelmap
    src_db = DatasetBackbone(src_db_path)
    labelmap = json.load(open(labelmap_path, 'r'))
    all_categories = src_db.query_all_category_names()

    instance_ids = []
    label_ids = []
    label_names = []
    image_paths = []
    img_widths = []
    img_heights = []

    for category_name in tqdm(all_categories):
        annotations = src_db.query_anno_info_by_category_name(category_name)
        if not annotations:
            print("Skip empty item: {} which has no annotations".format(category_name))
            continue
        for anno in annotations:
            inst_id = anno['id']
            label_name = anno['category']
            if label_name is None or label_name == '':
                print("Skip id:{} item: which has no label name".format(inst_id))
                continue
            img_info = src_db.query_img_info_by_filename(anno['filename'])[0]
            label_name = labelmap[label_name]['label_name']
            label_id = labelmap[label_name]['label_int']
            filename = anno['filename']
            width = img_info['w']
            height = img_info['h']
            image_path = src_db.query_img_abspath_by_filename(filename)[0]
            # push
            instance_ids.append(inst_id)
            label_ids.append(label_id)
            label_names.append(label_name)
            image_paths.append(image_path)
            img_widths.append(width)
            img_heights.append(height)
    data_dict = {
        'instance_id': instance_ids,
        'label_id': label_ids,
        'label_name': label_names,
        'image_path': image_paths,
        'width': img_widths,
        'height': img_heights,
    }

    data_df = pd.DataFrame(data_dict)
    data_df.to_csv('{}.csv'.format(args.output_dir))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--data_dir', type=str, default=None,
                        help='Path to source dataset backbone dir.')
    parser.add_argument('-od', '--output_dir', type=str, default=None,
                        help='Path to the output CSV filename.')
    parser.add_argument('-t', '--type', type=str, default=None,
                        help='')
    parser.add_argument('-l', '--labelmap', type=str, default=None,
                        help='')
    args = parser.parse_args()
    main(args)
