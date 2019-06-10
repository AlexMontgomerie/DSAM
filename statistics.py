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

TEST_SIZE=10

# model parameters
model_path     = 'model/alexnet.prototxt'
model_path     = 'model/vgg16.prototxt'
#model_path     = 'model/lenet.prototxt'
data_path_root = 'data/imagenet'
#data_path_root = 'data/mnist'
weights_path   = 'weight/alexnet.caffemodel'
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

print(random_data_files)

# save values for each layer
pixels = {}

# run network
print("Running Network... ")
for f in random_data_files:
    run_net(net,f)
    # store data
    for layer in net.blobs:
        layer_type = re.match("[a-z]+",str(layer))
        layer_type = layer_type.group(0)
        if layer_type=='conv' or layer_type=='pool' or layer_type=='data':
            if layer in pixels:
                pixels[layer] = np.concatenate( [ pixels[layer], layer_to_stream(net.blobs[layer].data[0][...]) ] )
            else:
                pixels[layer] = layer_to_stream(net.blobs[layer].data[0][...])

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

CORR_SIZE=1000

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

def hamming_distance(x1,x2):
    dist = x1 ^ x2
    return bin(dist).count('1')

def hamming_distance_stream(x1,x2):
    xor = np.bitwise_xor(x1,x2)
    f = lambda x : bin(x).count('1') # hamming distance
    vf = np.vectorize(f)
    #return [hamming_distance(stream[i],stream[i-1])/FIXED_WIDTH for i in range(1,len(stream))]
    return vf(xor)


print("Gathering Statistics... (hamming distance) ")
hamm_dist = {}
for layer in pixels:
    tmp = pixels[layer]
    hamm_dist[layer] = [ np.mean(hamming_distance_stream(tmp[i:],tmp[:-i])) for i in range(1,CORR_SIZE) ]

i=1
for layer in hamm_dist:
    plt.subplot(len(hamm_dist),1,i)
    if i == 1:
        plt.title("Hamming Distance against Offset")
    plt.plot(np.arange(1,CORR_SIZE),hamm_dist[layer])
    plt.ylabel(layer)
    if i != len(hamm_dist):
        plt.xticks([])
    i+=1
plt.xlabel('Offset, k')
#plt.xticks([])
plt.show()


