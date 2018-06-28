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

    search_image_id = 3

    with open(args.pkl, 'rb') as f:
        dataset = pickle.load(f)

    print('===== train =============')
    train_sentence_dataset = dataset['sentences']['train']
    train_image_dataset = dataset['images']['train']
    for i, image_ids in train_image_dataset.items():
        for j, image_id in enumerate(image_ids):
            if image_id == search_image_id:
                print('image_id: ', str(image_id))
                print('sentence is ...')
                print(train_sentence_dataset[i][j])

    print('===== restval =============')
    restval_sentence_dataset = dataset['sentences']['restval']
    restval_image_dataset = dataset['images']['restval']
    for i, image_ids in restval_image_dataset.items():
        for j, image_id in enumerate(image_ids):
            if image_id == search_image_id:
                print('image_id: ', str(image_id))
                print('sentence is ...')
                print(restval_sentence_dataset[i][j])

    print('===== val =============')
    val_sentence_dataset = dataset['sentences']['val']
    val_image_dataset = dataset['images']['val']
    for i, image_ids in val_image_dataset.items():
        for j, image_id in enumerate(image_ids):
            if image_id == search_image_id:
                print('image_id: ', str(image_id))
                print('sentence is ...')
                print(val_sentence_dataset[i][j])

    print('===== test =============')
    test_sentence_dataset = dataset['sentences']['test']
    test_image_dataset = dataset['images']['test']
    for i, image_ids in test_image_dataset.items():
        for j, image_id in enumerate(image_ids):
            if image_id == search_image_id:
                print('image_id: ', str(image_id))
                print('sentence is ...')
                print(test_sentence_dataset[i][j])



