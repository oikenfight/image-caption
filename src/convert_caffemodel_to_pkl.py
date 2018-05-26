import pickle as pickle
from chainer.links import caffe
import sys

model_path = sys.argv[1]
pkl_path = sys.argv[2]

model = caffe.CaffeFunction(model_path)

pickle.dump(model, open(pkl_path, "wb"))
