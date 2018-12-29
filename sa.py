import numpy as np
import sys, os, getopt
import caffe
import json
import copy
import tqdm
import math
from PIL import Image
import matplotlib.pyplot as plt

FIXED_WIDTH     = 16
FIXED_INT_SIZE  = 4

def hamming_distance(x1,x2):
    dist = x1 ^ x2
    return bin(dist).count('1')

def plot_image(filepath):
    # get image from filepath
    im = Image.open(filepath)
    # save as numpy array
    im = np.array(im,dtype=np.float32)

    print()

    im_folded = []

    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            for channel in range(im.shape[2]):
                im_folded.append(int(im[row,col,channel]))
    #print(im_folded)
    #plt.plot(im_folded[:500])

    im_sa = [hamming_distance(im_folded[i],im_folded[i-1]) for i in range(1,len(im_folded))]

    plt.plot(im_sa)

    plt.show()

def entropy(p_arr,bits):
    h = 0
    for p in p_arr:
        if p == 0:
            pass
        else:
            h -= p*math.log(p,2) + (1-p)*math.log(1-p,2)
    return h

# get the switching activity for a layer
def get_sa_layer(layer,bit):
    size = 0
    bit_val_prev = 0
    sa = 0
    # iterate: channels,width,rows
    if len(layer.shape) == 4:
        size = layer.shape[1]*layer.shape[2]*layer.shape[3]
        for row in range(layer.shape[3]):
            for col in range(layer.shape[2]):
                for channel in range(layer.shape[1]):
                    # get value of bit
                    bit_val = get_bit(ap_fixed(layer[0][channel][row][col],FIXED_WIDTH,FIXED_INT_SIZE),bit)
                    if bit_val != bit_val_prev:
                        sa += 1
                    bit_val_prev = bit_val
    else:
        return
    return sa/size

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
    # save as numpy array
    in_ = np.array(im,dtype=np.float32)
    # save each channel of input to network
    print(in_.shape)
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
    layer_sa = [ 0 for i in range(FIXED_WIDTH) ]
    for bit in range(FIXED_WIDTH):
        layer_sa[bit] = get_sa_layer(layer,bit)
    print(layer_sa)

#Run analysis across the whole network
def analyse_net(net):
    for layer in net.blobs:
        analyse_layer(net.blobs[layer].data[...])
        #print(net.blobs[layer].data[...].shape)
        #print(get_sa_layer(net.blobs[layer].data[...],1))

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
    #main(sys.argv[1:])
