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
from encoding import *
import scipy.stats
import matplotlib.pyplot as plt

TEST_SIZE=50

# model parameters
#model_path     = 'model/alexnet.prototxt'
model_path     = 'model/vgg16.prototxt'
#model_path     = 'model/lenet.prototxt'
data_path_root = 'data/imagenet'
#data_path_root = 'data/mnist'
#weights_path   = 'weight/alexnet.caffemodel'
weights_path   = 'weight/vgg16.caffemodel'
#weights_path   = 'weight/lenet.caffemodel'

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
print("Running Network... ")
for f in random_data_files:
    #run_net(net,f)
    # store data
    for layer in net.blobs:
        layer_type = re.match("[a-z]+",str(layer))
        layer_type = layer_type.group(0)
        if layer_type=='conv' or layer_type=='pool' or layer_type=='data':
            if layer in pixels:
                print("{}: {}".format(layer,net.blobs[layer].data.shape))
                #pixels[layer] = np.concatenate( [ pixels[layer], layer_to_stream(net.blobs[layer].data[...]) ] )
            #else:
                #pixels[layer] = layer_to_stream(net.blobs[layer].data[...])

print("Getting Average Switching Activity...")
base_sa     = {}
base_sa_var = {}
for layer in pixels:
    base_sa[layer] = get_sa_stream_avg(pixels[layer])
    base_sa_var[layer] = math.sqrt(get_sa_stream_var(pixels[layer]))
    print("{layer} switching activity: \t {sa} (sd={var})".format(layer=layer, sa=base_sa[layer], var=base_sa_var[layer]) )

'''
plt.hist(
    pixels['data'],
    density=True, histtype='stepfilled', alpha=0.2
)
plt.show()
'''
def autocorr(x, t=1):
    return np.corrcoef(np.array([x[:-t], x[t:]]))

CORR_SIZE=200

def bitwise(stream,shift=0):
    stream_out = np.bitwise_and(stream,(1<<shift))
    return 2*(stream_out/(2**shift))-1
'''
print("Gathering Statistics... (correlation) ")
correlation = {}
for layer in pixels:
    correlation[layer] = [ abs(autocorr(pixels[layer], i)[0][1]) for i in range(1,CORR_SIZE) ]

i=1
for layer in correlation:
    plt.subplot(len(correlation),1,i)
    if i == 1:
        plt.title("Correlation against Offset")
    plt.plot(np.arange(1,CORR_SIZE),correlation[layer])
    plt.ylabel(layer)
    if i != len(correlation):
        plt.xticks([])
    i+=1
plt.xlabel('offset, k')
plt.show()
'''
print("Gathering Statistics... (distance) ")
dist = {}
for layer in pixels:
    tmp = pixels[layer]
    dist[layer] = [np.linalg.norm(np.subtract(tmp[i:], tmp[:-i]))/len(tmp[i:]) for i in range(1,CORR_SIZE) ]

i=1
for layer in dist:
    plt.subplot(len(dist),1,i)
    if i == 1:
        plt.title("L2 Norm against Offset")
    plt.plot(np.arange(1,CORR_SIZE),dist[layer])
    plt.ylabel(layer)
    if i != len(dist):
        plt.xticks([])
    i+=1
plt.xlabel('offset, k')
#plt.xticks([])
plt.show()


'''
# encode pixels
pixels_encoded = {}
offset = {
  "data"  : 3,
  "conv1" : 96,
  "pool1" : 96,
  "conv2" : 256,
  "pool2" : 256,
  "conv3" : 384,
  "conv4" : 384,
  "conv5" : 256,
  "pool5" : 256
}
for layer in pixels:
    pixels_encoded[layer] = differential_encoding_stream( pixels[layer] , offset[layer])

print("Gathering Statistics... ")
for layer in pixels_encoded:
    #plt.acorr( pixels[layer], maxlags=100, label=layer)
    #plt.stem( [i for i in range(1,CORR_SIZE)], [ autocorr(pixels[layer], i)[0][1] for i in range(1,CORR_SIZE) ], label=layer)
    # tmp = bitwise(pixels[layer], 8, 8)
    tmp = pixels_encoded[layer]
    #tmp = bitwise(pixels[layer])
    idx = [i for i in range(1,CORR_SIZE)]
    #acorr = [ autocorr(pixels[layer], i)[0][1] for i in range(1,CORR_SIZE) ]
    acorr = [ autocorr(tmp, i)[0][1] for i in range(1,CORR_SIZE) ]
    print("Max Auto-Correlation ({layer}) \t = {max}, \t index = {index}".format(layer=layer,max=max(acorr),index=acorr.index(max(acorr))+1 ))
    #plt.show()
'''

'''
print("\n\n")
for layer in pixels:
    acorr = [ autocorr(pixels[layer], i)[0][1] for i in range(1,CORR_SIZE) ]
    print("Max Auto-Correlation ({layer}) \t = {max}, \t index = {index}".format(layer=layer,max=max(acorr),index=acorr.index(max(acorr))+1 ))
    #plt.show()

'''
'''
print("Running Bitwise Correlation ... ")
for layer in pixels:
    corr_total = []
    for i in range(FIXED_WIDTH):
        tmp = bitwise(pixels[layer], i)
        idx = [i for i in range(1,CORR_SIZE)]
        acorr = [ autocorr(tmp, i)[0][1] for i in range(1,CORR_SIZE) ]
        corr_total.append( ( max(acorr) , acorr.index(max(acorr))+1 ) )
    print('{layer} = '.format(layer=layer), corr_total)
'''
'''
for layer in pixels:
    corr_total = [0 for i in range(CORR_SIZE-1)]
    for index in range(len(pixels[layer])-2*CORR_SIZE):
        acorr = [ autocorr(pixels[layer][index:index+2*CORR_SIZE], i)[0][1] for i in range(1,CORR_SIZE) ]
        for i in range(CORR_SIZE-1):
          corr_total[i] += acorr[i]
    for i in range(CORR_SIZE-1):
        corr_total[i] /= (len(pixels[layer])-2*CORR_SIZE)
    print('{layer} = '.format(layer=layer), corr_total)
'''

'''
n_bits = [16, 8, 4, 2, 1]
for n_bit in n_bits:
    n_blocks = int(FIXED_WIDTH/n_bits)

    for block_index in n_blocks
'''
