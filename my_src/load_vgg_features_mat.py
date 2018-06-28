# -*- coding: utf-8 -*-

import argparse
import pickle as pickle
import json
import numpy as np
import scipy.io
# from net import ImageCaption
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read vgg feature mat format')

    parser.add_argument('--mat', '-m', required=True, type=str, help='input feature mat file path')

    args = parser.parse_args()

    print(args.mat)

    image_dataset = scipy.io.loadmat(args.mat)

    images = image_dataset['feats'].transpose((1, 0))

    # print(image_dataset)

    print(image_dataset['__version__'])
    print(image_dataset['__globals__'])
    print(image_dataset['__header__'])
    print('feats shape')
    print(type(image_dataset['feats']))
    print(image_dataset['feats'].shape)
    print('transpose')
    print(images.shape)

    for key, image in enumerate(images):
        print()
        print(key)
        print(image.shape)
        print(type(image))
        print(image)
        time.sleep(1)
