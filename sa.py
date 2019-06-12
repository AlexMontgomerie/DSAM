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

FIXED_WIDTH     = 16
FIXED_INT_SIZE  = 4

# DATA TYPES
def fixed16(val):
    return (int(val*(2<<(FIXED_WIDTH-FIXED_INT_SIZE))))&((2<<FIXED_WIDTH)-1)

# LAYER FUNCTIONS
def layer_to_stream(layer,quantiser=fixed16):
    stream = np.ravel(layer,order='F')
    vf = np.vectorize(quantiser)
    return vf(stream)

# ANALYSIS
def entropy(p_arr,bits):
    h = 0
    for p in p_arr:
        if p == 0:
            pass
        else:
            h -= p*math.log(p,2) + (1-p)*math.log(1-p,2)
    return h

def hamming_distance(x1,x2):
    dist = x1 ^ x2
    return bin(dist).count('1')

def hamming_distance_stream(x1,x2):
    xor = np.bitwise_xor(x1,x2)
    f = lambda x : bin(x).count('1') # hamming distance
    vf = np.vectorize(f)
    return vf(xor)

def get_sa_stream(stream):
    xor = np.bitwise_xor(stream[1:],stream[:-1]) # hamming distance (1)
    f = lambda x : bin(x).count('1')/FIXED_WIDTH # hamming distance (2)
    vf = np.vectorize(f)
    #return [hamming_distance(stream[i],stream[i-1])/FIXED_WIDTH for i in range(1,len(stream))]
    return vf(xor)

def get_sa_stream_avg(stream):
    sa_stream = get_sa_stream(stream)
    return np.mean(sa_stream)

def get_sa_stream_var(stream):
    sa_stream = get_sa_stream(stream)
    return np.var(sa_stream)

def num_ones_in_word(word):
    val = 0
    for i in range(8):
        val += (word >> i) & 1
    return val

# Run network
def run_net(net,filepath,scale=256):
    # get image from filepath
    im = Image.open(filepath)
    # resize image
    shape = net.blobs['data'].data[...].shape
    im = im.resize((shape[2],shape[3]),Image.ANTIALIAS)
    # save as numpy array
    in_ = np.array(im,dtype=np.float32)
    # normalise array
    in_ = np.true_divide(in_, scale)
    # save each channel of input to network
    if len(in_.shape) == 2:
        net.blobs['data'].data[...][0] = copy.deepcopy(np.array(in_,dtype=np.float32))
    else:
        for channel in range(in_.shape[2]):
            net.blobs['data'].data[...][0][channel] = copy.deepcopy(np.array(in_[:,:,channel],dtype=np.float32))
    # run network
    net.forward()
    # return network
    return net
