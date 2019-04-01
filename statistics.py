import numpy as np
import sys, os, getopt
import caffe
import json
import copy
import tqdm
import math
from PIL import Image
import matplotlib.pyplot as plt
import re
import random
from sa import *
import scipy.stats
import matplotlib.pyplot as plt

TEST_SIZE=10

# model parameters
model_path     = 'model/lenet.prototxt'
data_path_root = 'data/mnist'
weights_path   = 'weight/lenet.caffemodel'

# Initialise Network
net = caffe.Classifier(model_path,weights_path)

# get all Images
data_files = []
index=0
for (dirpath, dirnames, filenames) in os.walk(data_path_root):
    for filename in filenames:
        data_files.append(dirpath+'/'+filename)


random_data_files = [ random.choice(data_files) for x in range(TEST_SIZE) ]

# save values for each layer
pixels = {

}

# run network
for f in random_data_files:
    run_net(net,f)
    # store data
    for layer in net.blobs:
        layer_type = re.match("[a-z]+",str(layer))
        layer_type = layer_type.group(0)
        if layer_type=='conv' or layer_type=='pool' or layer_type=='data':
            if layer in pixels:
                pixels[layer] = np.concatenate( [ pixels[layer], layer_to_stream(net.blobs[layer].data[...], quantiser=float) ] )
            else:
                pixels[layer] = layer_to_stream(net.blobs[layer].data[...], quantiser=float)

print(pixels)

'''
plt.hist(
    pixels['data'],
    density=True, histtype='stepfilled', alpha=0.2
)
plt.show()
'''

for layer in pixels:
    plt.acorr( pixels[layer], maxlags=100)
    plt.show()
