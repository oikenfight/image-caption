import argparse
import numpy as np
import json
import os
import pickle as pickle
from image_model import VGG19
import time
import chainer
from chainer import Variable, serializers, cuda, functions as F


train_image_dir = './images/coco2014/train2014/'
val_image_dir = './images/coco2014/val2014/'
image_model = VGG19()


def load_model(model_path: str):
    """
    :param str model_path: this is model path
    :return:
    """
    image_model.load(model_path)
    return


def get_image_feature(filename: str):
    """
    :param str filename: this is image path
    :return numpy.ndarray features:
    """
    file_path = ''
    if os.path.exists(train_image_dir + filename):
        file_path = train_image_dir + filename
    elif os.path.exists(val_image_dir + filename):
        file_path = val_image_dir + filename
    else:
        print('============================')
        print('image file not found.')
        print(filename)
        exit()
        print('============================')

    feature = image_model.feature(file_path)
    return feature.data[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Feature pkl file')

    parser.add_argument('--train_json_path', '-t', required=True, type=str, help='input train stair caption json path')
    parser.add_argument('--val_json_path', '-v', required=True, type=str, help='input val stair caption json path')
    parser.add_argument('--model', '-m', required=True, type=str, help='input image model path')
    parser.add_argument('--output', '-o', required=True, type=str, help='output file name')

    args = parser.parse_args()

    features = {}

    # image model を load して画像特徴量を取得する準備をする
    load_model(args.model)

    print('====== train ====================')
    with open(args.train_json_path, 'r') as json_file:
        train_dataset = json.load(json_file)

    # 画像特徴量を算出
    for image_data in train_dataset['images']:
        image_id = image_data['id']
        feature = get_image_feature(image_data['file_name'])
        features[image_id] = np.array(feature)
        print(len(features))

    print()
    print('====== val ====================')
    with open(args.val_json_path, 'r') as json_file:
        val_dataset = json.load(json_file)

    # 画像特徴量を算出
    for image_data in val_dataset['images']:
        image_id = image_data['id']
        feature = get_image_feature(image_data['file_name'])
        features[image_id] = feature
        print(len(features))


    # dump to pkl
    with open(args.output, 'wb') as f:
        pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)


