# -*- coding: utf-8 -*-

import argparse
import pickle as pickle
import json
import numpy as np
import scipy.io
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read vgg feature mat format')

    parser.add_argument('--pkl', '-p', required=True, type=str, help='input dataset pkl file path')
    parser.add_argument('--number', '-n', required=True, type=str, help='how many read do you want to read')

    args = parser.parse_args()

    search_image_id = 1

    with open(args.pkl, 'rb') as f:
        dataset = pickle.load(f)

    for data_type in dataset['annotations']:
        print(data_type)

    print('===== train =============')
    train_annotation_dataset = dataset['annotations']['train']
    train_image_dataset = dataset['images']['train']
    for i, image_ids in train_image_dataset.items():
        print(image_ids)
        for j, image_id in enumerate(image_ids):
            if image_id == search_image_id:
                print('image_id: ', str(image_id))
                print('annotation is ...')
                print(train_annotation_dataset[i][j])

    print('===== val =============')
    val_annotation_dataset = dataset['annotations']['val']
    val_image_dataset = dataset['images']['val']
    for i, image_ids in val_image_dataset.items():
        for j, image_id in enumerate(image_ids):
            if image_id == search_image_id:
                print('image_id: ', str(image_id))
                print('annotation is ...')
                print(val_annotation_dataset[i][j])




