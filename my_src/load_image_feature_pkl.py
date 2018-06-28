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

    args = parser.parse_args()

    with open(args.pkl, 'rb') as f:
        dataset = pickle.load(f)

    # print(dataset)
    print(len(dataset))

    for image_id, feature in dataset.items():
        print(image_id)
        print(feature)
        time.sleep(1)




