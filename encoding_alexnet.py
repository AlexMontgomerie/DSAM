from encoding import *
import random

TEST_SIZE=10

net_name = 'alexnet'

if net_name == 'lenet':
    model_path     = 'model/lenet.prototxt'
    data_path_root = 'data/mnist'
    weights_path   = 'weight/lenet.caffemodel'

if net_name == 'alexnet':
    model_path     = 'model/alexnet.prototxt'
    data_path_root = 'data/imagenet'
    weights_path   = 'weight/alexnet.caffemodel'

if net_name == 'vgg':
    model_path     = 'model/vgg16.prototxt'
    data_path_root = 'data/imagenet'
    weights_path   = 'weight/vgg16.caffemodel'

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
print("RUNNING NETWORK")
for f in random_data_files:
    run_net(net,f)
    # store data
    for layer in net.blobs:
        layer_type = re.match("[a-z]+",str(layer))
        layer_type = layer_type.group(0)
        if layer_type=='conv' or layer_type=='pool' or layer_type=='data':
            print(net.blobs[layer].data.shape)
            if layer in pixels:
                pixels[layer] = np.concatenate( [ pixels[layer], layer_to_stream(net.blobs[layer].data[0][...] ) ] )
            else:
                pixels[layer] = layer_to_stream(net.blobs[layer].data[0][...])

# get baseline switching activity
base_sa = {}
print("BASELINE SA")
for layer in pixels:
    base_sa[layer] = get_sa_stream_avg(pixels[layer])
    print("{layer} switching activity: \t {sa}".format(layer=layer, sa=base_sa[layer]) )


# bus-invert encoding
print("BUS INVERT")
bus_invert_encoded = {}
for layer in pixels:
    bus_invert_encoded[layer] = bus_invert_stream( pixels[layer] )
    sa_avg = get_sa_stream_avg(bus_invert_encoded[layer])
    print("{layer} switching activity (bus invert): \t {sa}, reduction = {reduction}".format(
        layer=layer, sa=sa_avg, reduction= (sa_avg/base_sa[layer])*100 ) )


# adaptive encoding (static)
print("ADAPTIVE ENCODING STATIC")
pixels_adaptive_encoding_static = {}
for layer in pixels:
    pixels_adaptive_encoding_static[layer], code_table = adaptive_encoding_static_stream( pixels[layer] )
    sa_avg = get_sa_stream_avg(pixels_adaptive_encoding_static[layer])
    print("{layer} switching activity (adaptive encoding, static): \t {sa} \t (size={size}), reduction = {reduction}".format(
        layer=layer, sa=sa_avg, size=len(code_table), reduction= (sa_avg/base_sa[layer])*100 ) )


# load offset
if net_name == 'lenet':
    with open('coef/lenet.json', 'r') as f:
        offset = json.load(f)
if net_name == 'alexnet':
    with open('coef/alexnet.json', 'r') as f:
        offset = json.load(f)
if net_name == 'vgg':
    with open('coef/vgg.json', 'r') as f:
        offset = json.load(f)
        
# differential encoding
print("CSAM ENCODING")
csam_encoding = {}
for layer in pixels:
    csam_encoding[layer] = csam_encoding_stream( pixels[layer] , offset[layer])
    sa_avg = get_sa_stream_avg(csam_encoding[layer])
    print("{layer} switching activity (csam): \t {sa}, reduction = {reduction}".format(
        layer=layer, sa=sa_avg, reduction = (sa_avg/base_sa[layer])*100 ) )

print("DSAM ENCODING")
dsam_encoding = {}
for layer in pixels:
    dsam_encoding[layer], _ = dsam_encoding_stream( pixels[layer] , offset[layer])
    sa_avg = get_sa_stream_avg(dsam_encoding[layer])
    print("{layer} switching activity (dsam): \t {sa}, reduction = {reduction}".format(
        layer=layer, sa=sa_avg, reduction = (sa_avg/base_sa[layer])*100 ) )
    #perform decoding
