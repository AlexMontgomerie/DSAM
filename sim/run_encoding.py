import random
import sys
import getopt
#sys.path.append("..")
import os
import tqdm
os.environ['GLOG_minloglevel'] = '3' 
from src.sa import *
from src.encoding import *

def main(argv):

    # function variables 
    model_path    = ''
    data_path     = ''
    weights_path  = ''
    offset_path   = ''
    TEST_SIZE     = 1
    coding_scheme = 'base'

    try:
        opts,args = getopt.getopt(argv,"hm:d:w:n:o:",
          ['baseline','dsam','csam'])
    except getopt.GetoptError:
        sys.exit(2)
    for opt,arg in opts:
        if opt == '-h':
            print("usage : ")
            print("     python3 run_encoding.py -m <model path> -d <data path> -w <weight path> -n <number of images> -o <offset file>")
            print("         --base = baseline")
            print("         --binv = Bus Invert coding scheme")
            print("         --dsam = DSAM coding scheme")
            print("         --csam = CSAM coding scheme")
            print("         --apbm = APBM coding scheme")
            sys.exit()
        elif opt in ('-m'):
            model_path = arg
        elif opt in ('-d'):
            data_path  = arg
        elif opt in ('-w'):
            weights_path = arg
        elif opt in ('-o'):
            offset_path = arg
        elif opt in ('-n'):
            TEST_SIZE = int(arg)
        elif opt in ('--base'):
            coding_scheme = 'base'
        elif opt in ('--dsam'):
            coding_scheme = 'dsam'
        elif opt in ('--csam'):
            coding_scheme = 'csam'

    # Initialise Network
    net = caffe.Classifier(model_path,weights_path)

    # get all Images
    data_files = []
    index=0
    for (dirpath, dirnames, filenames) in os.walk(data_path):
        for filename in filenames:
            data_files.append(dirpath+'/'+filename)

    # choose random files
    random_data_files = [ random.choice(data_files) for x in range(TEST_SIZE) ]

    # save values for each layer
    pixels = {}
    
    # run network
    print("RUNNING NETWORK")
    for f in tqdm.tqdm(random_data_files):
        run_net(net,f)
        # store data
        for layer in net.blobs:
            layer_type = re.match("[a-z]+",str(layer))
            layer_type = layer_type.group(0)
            if layer_type=='conv' or layer_type=='pool' or layer_type=='data':
                if layer in pixels:
                    pixels[layer] = np.concatenate( [ pixels[layer], layer_to_stream(net.blobs[layer].data[0][...] ) ] )
                else:
                    pixels[layer] = layer_to_stream(net.blobs[layer].data[0][...])
    # base sa
    base_sa = {}
    for layer in pixels:
        base_sa[layer] = get_sa_stream_avg(pixels[layer])

    # coding scheme
    coding = {}
    for layer in pixels:
        # BASELINE
        if coding_scheme == 'base':
            coding[layer] = pixels[layer]
        # BI
        if coding_scheme == 'binv':
            coding[layer] = bus_invert_stream( pixels[layer] )
        # DSAM
        elif coding_scheme == 'dsam':
            with open(offset_path, 'r') as f:
                offset = json.load(f)
            coding[layer], sign = dsam_encoding_stream( pixels[layer] , offset[layer])
        # CSAM
        elif coding_scheme == 'csam':
            with open(offset_path, 'r') as f:
                offset = json.load(f)
            coding[layer] = csam_encoding_stream( pixels[layer] , offset[layer])
        # APBM
        elif coding_scheme == 'apbm':
            coding[layer], _ = adaptive_encoding_static_stream( pixels[layer] )
        sa_avg = get_sa_stream_avg(coding[layer])
        print("{layer} \t switching activity: \t {sa}, \t reduction = {reduction}".format(layer=layer, sa=round(sa_avg,4), reduction=round((base_sa[layer]-sa_avg)/base_sa[layer]*100,4) ))

if __name__ == "__main__":
    main(sys.argv[1:])
  
