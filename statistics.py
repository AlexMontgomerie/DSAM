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
pixels = {}

# run network
for f in random_data_files:
    run_net(net,f)
    # store data
    for layer in net.blobs:
        layer_type = re.match("[a-z]+",str(layer))
        layer_type = layer_type.group(0)
        if layer_type=='conv' or layer_type=='pool' or layer_type=='data':
            if layer in pixels:
                pixels[layer] = np.concatenate( [ pixels[layer], layer_to_stream(net.blobs[layer].data[...]) ] )
            else:
                pixels[layer] = layer_to_stream(net.blobs[layer].data[...])

'''
plt.hist(
    pixels['data'],
    density=True, histtype='stepfilled', alpha=0.2
)
plt.show()
'''
def autocorr(x, t=1):
    return np.corrcoef(np.array([x[:-t], x[t:]]))

CORR_SIZE=100

def bitwise(stream,size=16,shift=0):
    mask = (2**size) - 1
    stream_out = []
    for val in stream:
        #stream_out.append( ( ( val & ( mask << shift ) ) >> shift ) & ((2**FIXED_WIDTH)-1) )
        stream_out.append( ( ( val & ( mask << shift ) ) ) & ((2**FIXED_WIDTH)-1) )
    return stream_out

'''
for layer in pixels:
    #plt.acorr( pixels[layer], maxlags=100, label=layer)
    #plt.stem( [i for i in range(1,CORR_SIZE)], [ autocorr(pixels[layer], i)[0][1] for i in range(1,CORR_SIZE) ], label=layer)
    tmp = bitwise(pixels[layer], 8, 8)
    #tmp = bitwise(pixels[layer])
    idx = [i for i in range(1,CORR_SIZE)]
    #acorr = [ autocorr(pixels[layer], i)[0][1] for i in range(1,CORR_SIZE) ]
    acorr = [ autocorr(tmp, i)[0][1] for i in range(1,CORR_SIZE) ]
    print("Max Auto-Correlation ({layer}) \t = {max}, \t index = {index}".format(layer=layer,max=max(acorr),index=acorr.index(max(acorr))+1 ))
    #plt.show()

print("\n\n")
for layer in pixels:
    acorr = [ autocorr(pixels[layer], i)[0][1] for i in range(1,CORR_SIZE) ]
    print("Max Auto-Correlation ({layer}) \t = {max}, \t index = {index}".format(layer=layer,max=max(acorr),index=acorr.index(max(acorr))+1 ))
    #plt.show()
'''

for layer in pixels:
    corr_total = []
    for i in range(FIXED_WIDTH):
        tmp = bitwise(pixels[layer], 1, i)
        idx = [i for i in range(1,CORR_SIZE)]
        acorr = [ autocorr(tmp, i)[0][1] for i in range(1,CORR_SIZE) ]
        corr_total.append( ( max(acorr) , acorr.index(max(acorr))+1 ) )
    print('{layer} = '.format(layer=layer), corr_total)

'''
n_bits = [16, 8, 4, 2, 1]
for n_bit in n_bits:
    n_blocks = int(FIXED_WIDTH/n_bits)

    for block_index in n_blocks
'''
