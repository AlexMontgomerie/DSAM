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
'''
def layer_to_stream(layer,quantiser=fixed16):
    stream = []
    for row in range(layer.shape[3]):
        for col in range(layer.shape[2]):
            for channel in range(layer.shape[1]):
                stream.append(quantiser(layer[0][channel][row][col]))
    return np.array(stream)
'''
def layer_to_stream(layer,quantiser=fixed16):
    stream = []
    for row in range(layer.shape[3]):
        for col in range(layer.shape[2]):
            for channel in range(layer.shape[1]):
                stream.append( layer[0][channel][row][col] )
    return np.array(stream)

# ENCODING METHODS

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

def get_sa_stream(stream):
    return [hamming_distance(stream[i],stream[i-1])/FIXED_WIDTH for i in range(1,len(stream))]

def get_sa_stream_avg(stream):
    sa_stream = get_sa_stream(stream)
    return (sum(sa_stream)/float(len(sa_stream)+1))

def num_ones_in_word(word):
    val = 0
    for i in range(8):
        val += (word >> i) & 1
    return val

# Get Bit of Fixed Point
def get_bit(val,bit):
    return (val >> bit) & 1

# Fixed Point
def ap_fixed(val,width,int_size):
    return (int(val*(2<<(width-int_size))))&((2<<width)-1)

# Run network
def run_net(net,filepath):
    # get image from filepath
    im = Image.open(filepath)
    # resize image
    shape = net.blobs['data'].data[...].shape
    im = im.resize((shape[2],shape[3]),Image.ANTIALIAS)
    # save as numpy array
    in_ = np.array(im,dtype=np.float32)
    # normalise array
    in_ = np.true_divide(in_, 256)
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

# Run analysis of layer of the network
def analyse_layer(layer):
    return get_sa_stream_avg(
        layer_to_stream(
            layer
        )
    )
    ##layer_sa = [ 0 for i in range(FIXED_WIDTH) ]
    #for bit in range(FIXED_WIDTH):
    #    layer_sa[bit] = get_sa_layer(layer,bit)
    #print(layer_sa)
    #return sum(layer_sa) / len(layer_sa)
    #return layer_sa

#Run analysis across the whole network
def analyse_net(net):
    layer_sa = []
    for layer in net.blobs:
        layer_type = re.match("[a-z]+",str(layer))
        layer_type = layer_type.group(0)
        if layer_type=='conv' or layer_type=='pool' or layer_type=='data':
            layer_sa.append( analyse_layer(net.blobs[layer].data[...]) )
        #print(net.blobs[layer].data[...].shape)
        #print(get_sa_layer(net.blobs[layer].data[...],1))
    print(layer_sa)
    return layer_sa

def main(argv):

    model_path    = ''
    data_path     = ''
    weights_path  = ''

    try:
        opts,args = getopt.getopt(argv,"hm:d:w:")
    except getopt.GetoptError:
        print('sa.py -m <model path> -d <data path> -w <weights path> ')
        sys.exit(2)
    for opt,arg in opts:
        if opt == '-h':
            print('sa.py -m <model path> -d <data path> -w <weights path>')
            sys.exit()
        elif opt in ('-m'):
            model_path = arg
        elif opt in ('-d'):
            data_path  = arg
        elif opt in ('-w'):
            weights_path = arg

    # Initialise Network
    net = caffe.Classifier(model_path,weights_path)

    # Run for given data
    run_net(net,data_path)

    # Analyse Network
    analyse_net(net)

if __name__=="__main__":
    #print(hamming_distance(3,0))
    plot_image('data/alexnet.jpg')
    main(sys.argv[1:])
