import numpy as np
import sys, os, getopt
import caffe
import json
import copy
import tqdm
import math
from PIL import Image

def get_bit(val,shift):
    return (val >> shift) & 1

def entropy(p_arr,bits):
    h = 0
    for p in p_arr:
        if p == 0:
            pass
        else:
            h -= p*math.log(p,2) + (1-p)*math.log(1-p,2)
    return h


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
    if len(in_.shape) == 2:
        net.blobs['data'].data[...][0] = copy.deepcopy(np.array(in_,dtype=np.float32))
    else: 
        for channel in in_.shape[2]:
            net.blobs['data'].data[...][0][channel] = copy.deepcopy(np.array(in_[:,:,channel],dtype=np.float32))
    # run network
    net.forward()
    # return network
    return net

# Run analysis of layer of the network
def analyse_layer(layer):
    pass

#Run analysis across the whole network
def analyse_net(net):
    for layer in net.blobs:
        analyse_layer(layer)

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
    main(sys.argv[1:])
