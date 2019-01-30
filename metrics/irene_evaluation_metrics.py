import argparse
import csv
import json
import os
import time
from shutil import copyfile

import numpy as np

'''
@param emb_array: 2d np array to search
       paths: a list to get image path
       labels: a list to get image label
       top_k: an int number of results to retrieve

=== QUERY AND DATABASE ===
Query: Product images, Database: User images; Mostly to see how centralized the user images are, relative to product images.
Query: User images, Database: Product images; Mostly to imitate the user scenario
Database...
    Only valiadation set: To see the behaviors the model learn in the unseen set
    Only training set: To see the behaviors the model learn in the seen set
    Training set + validation set: Use the biggest database to see how the model behaves

=== PID LEVEL ===
Top-1 pid accuracy: See the model's capability to retrieve the same product at the first rank.
Top-k pid recall: See the model's capability to retrieve the same product in the first k rank, i.e. count it right if correct prediction in top k.
Top-k pid mAP: See the model's capability to retrieve all or centain percentage of same product images from the database. Need to calculate by cat/supercat...

=== MID CATEGORY LEVEL ===
Top-k mid category accuracy: See the model's capability to retrieve same mid-category products in top-K.

=== SUPER CATEGORY LEVEL ===
### You can choose if you want to evaluate super category level accuracy because we can filter the retrieval results with detector's prediction.
Top-k super category accuracy: See the model's capability to retrieve same super-category products in top-K.

=== ATTRIBUTE LEVEL ===
### TODO ###

'''


def sort_distance_result(db_emb_array, que_emb, db_labels, top_k):
    diff = np.subtract(db_emb_array, que_emb)
    dist = np.sum(np.square(diff), 1)
    sort_dist_idx = np.argsort(dist)
    sort_dist_label = np.array(db_labels)[sort_dist_idx]
    top_dist_idx = sort_dist_idx[:top_k]
    top_dist_label = np.array(db_labels)[top_dist_idx]
    return top_dist_label, sort_dist_label


def calculate_pid_topk_recall(que_paths, que_emb_array, que_labels, db_paths, db_emb_array, db_labels, failed_dir,
                              top_k=5):
    same_retrieval = 0
    actual_same = 0
    for i, que_emb in enumerate(que_emb_array):
        que_label = que_labels[i]
        top_dist_label, _ = sort_distance_result(db_emb_array, que_emb, db_labels, top_k)
        # top 1
        if que_label == top_dist_label[0]:
            actual_same += 1
        # top k
        if que_label in top_dist_label:
            same_retrieval += 1
        else:
            dst_dir = os.path.join(failed_dir, que_paths[i].split('/')[4], que_paths[i].split('/')[5])
            if os.path.exists(dst_dir) != True:
                os.makedirs(dst_dir)
            dst = os.path.join(dst_dir, que_paths[i].split('/')[-1])
            copyfile(que_paths[i], dst)

    top1_acc = actual_same / float(len(que_emb_array))
    topk_recall = same_retrieval / float(len(que_emb_array))
    return top1_acc, topk_recall


def calculate_pid_topk_mAP(que_paths, que_emb_array, que_labels, db_paths, db_emb_array, db_labels, failed_dir,
                           recall=0.5):
    not_in_db = 0
    mAP = 0
    for i, que_emb in enumerate(que_emb_array):
        que_label = que_labels[i]
        _, sort_dist_label = sort_distance_result(db_emb_array, que_emb, db_labels, len(db_paths))

        k = 1
        prec_ls = []
        recall_ls = []
        flag = True
        start = time.time()
        while flag == True:
            top_dist_label = sort_dist_label[:k]
            tp = len(np.where(top_dist_label == que_label)[0])
            fp = len(np.where(top_dist_label != que_label)[0])
            fn = len(np.where(sort_dist_label == que_label)[0]) - tp
            if (tp + fn) != 0:
                precision = tp / float(tp + fp)
                recall = tp / float(tp + fn)
                # print ("precision={}, recall={}".format(precision, recall))
                prec_ls.append(precision)
                recall_ls.append(recall)
                k += 1
                if recall >= 0.5:
                    flag = False
            else:
                not_in_db += 1
                flag = False
        # calculate mAP
        mAP += mAP_metrics(recall_ls, prec_ls, recall)
    print('not in db: {}'.format(not_in_db))
    mAP = mAP / (len(que_emb_array) - not_in_db)
    return mAP


def calculate_cat_topk_acc(que_paths, que_emb_array, que_labels, db_paths, db_emb_array, db_labels, top_k=30):
    # labels in que_labels are target labels
    avg_acc = 0
    for i, que_emb in enumerate(que_emb_array):
        que_label = que_labels[i]
        top_dist_label, _ = sort_distance_result(db_emb_array, que_emb, db_labels, top_k)
        acc = len(np.where(top_dist_label == que_label)[0]) / top_k
        avg_acc += acc
    avg_acc /= float(len(que_emb_array))
    return avg_acc


def mAP_metrics(r_ls, p_ls, recall):
    max_pre = 0
    if r_ls != []:
        recalls = np.array(r_ls)
        precisions = np.array(p_ls)
        # for r_val in range(len(r_ls)+1):
        for r_val in range(int(recall / .1) + 1):
            r_thr = 0.1 * r_val
            max_pre += np.max(precisions[np.where(recalls >= r_thr)])
        ap = max_pre / (int(recall / .1) + 1)
    else:
        ap = 0
    return ap


def get_paths_labels(csv_file, midcat_map, supercat_map):
    nrof_skipped_img = 0
    paths = []
    labels = []
    sku_labels = []
    mid_labels = []
    super_labels = []
    with open(csv_file, 'r') as f_csv:
        rows = csv.DictReader(f_csv)
        for row in rows:
            path = row['cropped_filename']
            label = row['category_id'] + '-' + row['pid'] + '-' + row['brand_id']
            sku_label = row['category_id'] + '-' + row['pid']
            category_id = row['category_id']
            mid_label = midcat_map["midcat_map"][category_id]

            if category_id in supercat_map["table"]["1"]:
                super_label = 1
            elif category_id in supercat_map["table"]["2"]:
                super_label = 2
            elif category_id in supercat_map["table"]["3"]:
                super_label = 3

            if os.path.exists(path):
                paths.append(path)
                labels.append(label)
                sku_labels.append(sku_label)
                mid_labels.append(mid_label)
                super_labels.append(super_label)
            else:
                nrof_skipped_img += 1

    if nrof_skipped_img > 0:
        print('Skipped %d db images' % nrof_skipped_img)
    return paths, labels, sku_labels, mid_labels, super_labels


def get_map(mapfile):
    with open(mapfile) as f:
        mapfile = json.load(f)
    return mapfile


### Example ###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Config = ConfigParser.ConfigParser()

    parser.add_argument('--midcat_map', type=str, default='/volSSD03/tmall/tb_midcat_map.json')
    parser.add_argument('--supercat_map', type=str, default='/volSSD03/tmall/tmall_label_table.json')
    parser.add_argument('--que_csv', type=str,
                        default='/volSSD03/tmall/experiment_data/vis4_clustering_20933_val_user.csv')
    parser.add_argument('--db_csv', type=str,
                        default='/volSSD03/tmall/experiment_data/vis4_clustering_20933_val_pro.csv')
    parser.add_argument('--embeddings', type=str,
                        default='/volNAS/irene/facenet/data/train_tb_v1/20181022_073332_103020/small/embeddings.npz')
    parser.add_argument('--failed_dir', type=str, default='/volNAS/irene/facenet/failed_ex/modelname')
    parser.add_argument('--top1_acc', action='store_true')
    parser.add_argument('--top5_recall', action='store_true')
    parser.add_argument('--topk_midcatacc', action='store_true')
    parser.add_argument('--topk_supercatacc', action='store_true')
    parser.add_argument('--topk_mAP', action='store_true')
    args = parser.parse_args()

    # get paths and labels
    midcat_map = get_map(args.midcat_map)
    supercat_map = get_map(args.supercat_map)
    que_paths, que_labels, que_sku_labels, que_mid_labels, que_super_labels = get_paths_labels(args.que_csv, midcat_map,
                                                                                               supercat_map)
    db_paths, db_labels, db_sku_labels, db_mid_labels, db_super_labels = get_paths_labels(args.db_csv, midcat_map,
                                                                                          supercat_map)

    # load numpy array
    que_emb_array = np.load(args.embeddings)['que_emb_array']
    db_emb_array = np.load(args.embeddings)['db_emb_array']

    if args.top1_acc or args.top5_recall:
        top1_acc, top6_recall = calculate_pid_topk_recall(que_paths, que_emb_array, que_labels, db_paths, db_emb_array,
                                                          db_labels, args.failed_dir, top_k=6)
        top1_acc_sku, top6_recall_sku = calculate_pid_topk_recall(que_paths, que_emb_array, que_sku_labels, db_paths,
                                                                  db_emb_array, db_sku_labels, args.failed_dir, top_k=6)
        print('Top 1 BRAND_ID Acc: {}'.format(top1_acc))
        print('Top 1 SKU Acc: {}'.format(top1_acc_sku))
        print('Top 6 Recall BRAND_ID: {}'.format(top6_recall))
        print('Top 6 Recall SKU: {}'.format(top6_recall_sku))

    if args.topk_midcatacc and args.topk_supercatacc:
        avg_acc_mid = calculate_cat_topk_acc(que_paths, que_emb_array, que_mid_labels, db_paths, db_emb_array,
                                             db_mid_labels, top_k=10)
        avg_acc_super = calculate_cat_topk_acc(que_paths, que_emb_array, que_super_labels, db_paths, db_emb_array,
                                               db_super_labels, top_k=10)
        print('Top 10 Mid Cat Acc: {}'.format(avg_acc_mid))
        print('Top 10 Super Cat Acc: {}'.format(avg_acc_super))
    elif args.topk_midcatacc:
        avg_acc = calculate_cat_topk_acc(que_paths, que_emb_array, que_mid_labels, db_paths, db_emb_array,
                                         db_mid_labels, top_k=10)
        print('Top 10 Mid Cat Acc: {}'.format(avg_acc))
    elif args.topk_supercatacc:
        avg_acc = calculate_cat_topk_acc(que_paths, que_emb_array, que_super_labels, db_paths, db_emb_array,
                                         db_super_labels, top_k=10)
        print('Top 10 Super Cat Acc: {}'.format(avg_acc))

    if args.topk_mAP:
        mAP = calculate_pid_topk_mAP(que_paths, que_emb_array, que_sku_labels, db_paths, db_emb_array, db_sku_labels,
                                     args.failed_dir, recall=0.5)
        print('Top 10 SKU mAP: {}'.format(mAP))