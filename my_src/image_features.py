import argparse
import numpy as np
import pickle as pickle
from image_model import VGG19
import chainer
from chainer import Variable, serializers, cuda, functions as F


class ImageFeature:
    feature_num = 4096
    hidden_num = 512
    beam_width = 20
    max_length = 60
    image_model = VGG19()

    def load_model(self, model_path: str):
        """
        :param str model_path: this is model path
        :return:
        """
        self.image_model.load(model_path)
        return

    def features(self, image_path: str) -> list:
        """
        :param str image_path: this is image path
        :return list features:
        """
        return self.image_model.feature(image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract image features')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model', '-m', required=True, type=str, help='input image model file path (*.caffemodel or *.pkl)')
    parser.add_argument('--image', '-i', required=True, type=str, help='input image file path')

    args = parser.parse_args()

    image_feature = ImageFeature()

    image_feature.load_model(args.model)
    feature = image_feature.features(args.image)

    print(feature)
    print(feature.shape)
