# -*- coding: utf-8 -*-

import argparse
import pickle as pickle
import json
import numpy as np
import scipy.io
import random
import chainer
from chainer import cuda, optimizers, serializers, functions as F
from chainer.functions.evaluation import accuracy
from net import ImageCaption
import time

parser = argparse.ArgumentParser(description='Train image caption model')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--annotation', '-a', required=True, type=str,
                    help='input annotations dataset file path')
parser.add_argument('--image', '-i', required=True, type=str,
                    help='input images file path')
parser.add_argument('--model', '-m', default=None, type=str,
                    help='input model and state file path without extension')
parser.add_argument('--output', '-o', required=True, type=str,
                    help='output model and state file path without extension')
parser.add_argument('--iter', default=100, type=int,
                    help='output model and state file path without extension')
args = parser.parse_args()

gpu_device = None
args = parser.parse_args()
xp = np
if args.gpu >= 0:
    cuda.check_cuda_available()
    gpu_device = args.gpu
    cuda.get_device(gpu_device).use()
    xp = cuda.cupy

with open(args.annotation, 'rb') as f:
    annotation_dataset = pickle.load(f)

with open(args.image, 'rb') as f:
    images = pickle.load(f)
    
# image_dataset = scipy.io.loadmat(args.image)
# images = image_dataset['feats'].transpose((1, 0))

# for image in images:
#     print(type(image))
#     print(image)
#     print()
#     time.sleep(1)

train_image_ids = annotation_dataset['images']['train']
train_annotations = annotation_dataset['annotations']['train']
test_image_ids = annotation_dataset['images']['val']
test_annotations = annotation_dataset['annotations']['val']
word_ids = annotation_dataset['word_ids']
# feature_num = images.shape[1]
feature_num = 4096
hidden_num = 512
batch_size = 128

print('word count: ', len(word_ids))
caption_net = ImageCaption(len(word_ids), feature_num, hidden_num)
if gpu_device is not None:
    caption_net.to_gpu(gpu_device)
optimizer = optimizers.Adam()
optimizer.setup(caption_net)

if args.model is not None:
    serializers.load_hdf5(args.model + '.model', caption_net)
    serializers.load_hdf5(args.model + '.state', optimizer)

bos = word_ids['<S>']
eos = word_ids['</S>']
unknown = word_ids['<UNK>']


def random_batches(image_groups, annotation_groups):
    batches = []
    for image_ids, annotations in zip(image_groups, annotation_groups):
        length = len(annotations)
        index = np.arange(length, dtype=np.int32)
        np.random.shuffle(index)
        for n in range(0, length, batch_size):
            batch_index = index[n:n + batch_size]
            batches.append((image_ids[batch_index], annotations[batch_index]))
    random.shuffle(batches)
    return batches


def make_groups(image_ids, annotations, train=True):
    # 調査: boundaries 何？
    if train:
        boundaries = [1, 6, 11, 16, 21, 31, 41, 51]
    else:
        boundaries = range(1, 41)
    annotation_groups = []
    image_groups = []
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        size = sum(map(lambda x: len(annotations[x]), range(begin, end)))
        sub_annotations = np.full((size, end + 1), eos, dtype=np.int32)
        sub_annotations[:, 0] = bos
        sub_image_ids = np.zeros((size,), dtype=np.int32)
        offset = 0
        for n in range(begin, end):
            length = len(annotations[n])
            if length > 0:
                sub_annotations[offset:offset + length, 1:n + 1] = annotations[n]
                sub_image_ids[offset:offset + length] = image_ids[n]
            offset += length
        annotation_groups.append(sub_annotations)
        image_groups.append(sub_image_ids)
    return image_groups, annotation_groups


def forward(net, image_batch, annotation_batch, train=True):
    images = xp.asarray(image_batch)
    n, annotation_length = annotation_batch.shape
    net.initialize(images)
    loss = 0
    acc = 0
    size = 0
    for i in range(annotation_length - 1):
        target = xp.where(xp.asarray(annotation_batch[:, i]) != eos, 1, 0).astype(np.float32)
        if (target == 0).all():
            break
        with chainer.using_config('train', train):
            with chainer.using_config('enable_backprop', train):
                x = xp.asarray(annotation_batch[:, i])
                t = xp.asarray(annotation_batch[:, i + 1])
                y = net(x)
                y_max_index = xp.argmax(y.data, axis=1)
                mask = target.reshape((len(target), 1)).repeat(y.data.shape[1], axis=1)
                y = y * mask
                loss += F.softmax_cross_entropy(y, t)
                acc += xp.sum((y_max_index == t) * target)
                size += xp.sum(target)
    return loss / size, float(acc) / size, float(size)


def forward_image_batch(images, image_id_batch):
    image_batch = []
    for image_id in image_id_batch:
        image_batch.append(images[image_id])
    return np.array(image_batch)


def train(epoch_num):
    image_groups, annotation_groups = make_groups(train_image_ids, train_annotations)
    test_image_groups, test_annotation_groups = make_groups(test_image_ids, test_annotations, train=False)
    for epoch in range(epoch_num):
        batches = random_batches(image_groups, annotation_groups)
        sum_loss = 0
        sum_acc = 0
        sum_size = 0
        batch_num = len(batches)
        for i, (image_id_batch, annotation_batch) in enumerate(batches):
            # loss, acc, size = forward(caption_net, images[image_id_batch], annotation_batch)
            loss, acc, size = forward(caption_net, forward_image_batch(images, image_id_batch), annotation_batch)
            caption_net.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
            annotation_length = annotation_batch.shape[1]
            sum_loss += float(loss.data) * size
            sum_acc += acc * size
            sum_size += size
            if (i + 1) % 500 == 0:
                print('{} / {} loss: {} accuracy: {}'.format(i + 1, batch_num, sum_loss / sum_size, sum_acc / sum_size))
        print('epoch: {} done'.format(epoch + 1))
        print('train loss: {} accuracy: {}'.format(sum_loss / sum_size, sum_acc / sum_size))
        sum_loss = 0
        sum_acc = 0
        sum_size = 0
        for image_ids, annotations in zip(test_image_groups, test_annotation_groups):
            if len(annotations) == 0:
                continue
            size = len(annotations)
            for i in range(0, size, batch_size):
                image_id_batch = image_ids[i:i + batch_size]
                annotation_batch = annotations[i:i + batch_size]
                loss, acc, size = forward(caption_net, forward_image_batch(images, image_id_batch), annotation_batch, train=False)
                annotation_length = annotation_batch.shape[1]
                sum_loss += float(loss.data) * size
                sum_acc += acc * size
                sum_size += size
        print('test loss: {} accuracy: {}'.format(sum_loss / sum_size, sum_acc / sum_size))

        serializers.save_hdf5(args.output + '_{0:04d}.model'.format(epoch), caption_net)
        serializers.save_hdf5(args.output + '_{0:04d}.state'.format(epoch), optimizer)

train(args.iter)
