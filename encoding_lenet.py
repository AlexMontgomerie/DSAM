from encoding import *
import random

TEST_SIZE=5

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
                pixels[layer] = np.concatenate( [ pixels[layer], layer_to_stream(net.blobs[layer].data[...] ) ] )
            else:
                pixels[layer] = layer_to_stream(net.blobs[layer].data[...])

# get baseline switching activity
for layer in pixels:
    print("{layer} switching activity: \t {sa}".format(layer=layer, sa=get_sa_stream_avg(pixels[layer])) )

# gray encoding
pixels_gray_encoding = {}
for layer in pixels:
    pixels_gray_encoding[layer] = gray_encoding_stream( pixels[layer] )
    #print(pixels_gray_encoding[layer])
    print("{layer} switching activity (gray encoding): \t {sa}".format( layer=layer, sa=get_sa_stream_avg(pixels_gray_encoding[layer]) ) )

# adaptive encoding (static)
pixels_adaptive_encoding_static = {}
for layer in pixels:
    pixels_adaptive_encoding_static[layer] = adaptive_encoding_static_stream( pixels[layer] )
    #print(pixels_gray_encoding[layer])
    print("{layer} switching activity (adaptive encoding, static): \t {sa}".format( layer=layer, sa=get_sa_stream_avg(pixels_adaptive_encoding_static[layer]) ) )

# adaptive encoding
pixels_adaptive_encoding = {}
for layer in pixels:
    pixels_adaptive_encoding[layer] = adaptive_encoding_stream( pixels[layer] , 500 )
    #print(pixels_gray_encoding[layer])
    print("{layer} switching activity (adaptive encoding, 500): \t {sa}".format( layer=layer, sa=get_sa_stream_avg(pixels_adaptive_encoding[layer]) ) )

# differential encoding
pixels_differential_encoding = {}
for layer in pixels:
    pixels_differential_encoding[layer] = differential_encoding_stream( pixels[layer] , 20 )
    #print(pixels_gray_encoding[layer])
    print("{layer} switching activity (differential encoding): \t {sa}".format( layer=layer, sa=get_sa_stream_avg(pixels_differential_encoding[layer]) ) )
