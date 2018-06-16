import argparse
import numpy as np
import pickle as pickle
from image_model import VGG19
from net import ImageCaption
import chainer
from chainer import Variable, serializers, cuda, functions as F

feature_num = 4096
hidden_num = 512
beam_width = 20
max_length = 60


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract image features')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model', '-m', required=True, type=str, help='input image model file path (*.caffemodel or *.pkl)')
    parser.add_argument('--image', '-i', required=True, type=str, help='input image file path')

    args = parser.parse_args()

    image_model = VGG19()
    image_model.load(args.model)

    cuda.check_cuda_available()

    xp = np
    if args.gpu >= 0:
        print('using gpu')
        cuda.check_cuda_available()
        gpu_device = args.gpu
        cuda.get_device(gpu_device).use()
        xp = cuda.cupy
        image_model.to_gpu(gpu_device)
        caption_net.to_gpu(gpu_device)

    feature = image_model.feature(args.image)

    print(feature)
    print(feature.shape)
