from encoding import *
import random

random.seed(127232)

TEST_SIZE=100

model_path     = 'model/alexnet.prototxt'
data_path_root = 'data/imagenet'
weights_path   = 'weight/alexnet.caffemodel'

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
            if layer in pixels:
                pixels[layer] = np.concatenate( [ pixels[layer], layer_to_stream(net.blobs[layer].data[...] ) ] )
            else:
                pixels[layer] = layer_to_stream(net.blobs[layer].data[...])

# get baseline switching activity
print("BASELINE SA")
for layer in pixels:
    print("{layer} switching activity: \t {sa}".format(layer=layer, sa=get_sa_stream_avg(pixels[layer])) )

'''
# gray encoding
pixels_gray_encoding = {}
for layer in pixels:
    pixels_gray_encoding[layer] = gray_encoding_stream( pixels[layer] )
    #print(pixels_gray_encoding[layer])
    print("{layer} switching activity (gray encoding): \t {sa}".format( layer=layer, sa=get_sa_stream_avg(pixels_gray_encoding[layer]) ) )

'''
# adaptive encoding (static)
print("ADAPTIVE ENCODING STATIC")
pixels_adaptive_encoding_static = {}
for layer in pixels:
    pixels_adaptive_encoding_static[layer], code_table = adaptive_encoding_static_stream( pixels[layer] )
    #print(pixels_gray_encoding[layer])
    print("{layer} switching activity (adaptive encoding, static): \t {sa} \t (size={size})".format( layer=layer, sa=get_sa_stream_avg(pixels_adaptive_encoding_static[layer]), size=len(code_table) ) )

'''
# adaptive encoding
pixels_adaptive_encoding = {}
for layer in pixels:
    pixels_adaptive_encoding[layer], _ = adaptive_encoding_stream( pixels[layer] , 500 )
    #print(pixels_gray_encoding[layer])
    print("{layer} switching activity (adaptive encoding, 500): \t {sa}".format( layer=layer, sa=get_sa_stream_avg(pixels_adaptive_encoding[layer]) ) )

'''
# differential encoding
print("DIFFERENTIAL ENCODING")
pixels_differential_encoding = {}
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
    pixels_differential_encoding[layer] = differential_encoding_stream( pixels[layer] , offset[layer])
    #print(pixels_gray_encoding[layer])
    print("{layer} switching activity (differential encoding): \t {sa}".format( layer=layer, sa=get_sa_stream_avg(pixels_differential_encoding[layer]) ) )

print("DIFFERENTIAL ENCODING")
pixels_differential_encoding_2 = {}
offset = {
  "data"  : 1,
  "conv1" : 96,
  "pool1" : 192,
  "conv2" : 256,
  "pool2" : 512,
  "conv3" : 384,
  "conv4" : 384,
  "conv5" : 256,
  "pool5" : 256
}
for layer in pixels_differential_encoding:
    pixels_differential_encoding_2[layer] = differential_encoding_stream( pixels_differential_encoding[layer] , offset[layer])
    #print(pixels_gray_encoding[layer])
    print("{layer} switching activity (differential encoding): \t {sa}".format( layer=layer, sa=get_sa_stream_avg(pixels_differential_encoding_2[layer]) ) )


######################################################################

'''

block_sizes = [ 25, 50, 75, 100, 150, 200, 250, 500 ]
#block_sizes = [ 500, 1000 ]
switching_activity = {}
mem_size = {}

pixels = {
    'data' : pixels['data']
}

for block_size in block_sizes:
    print('BLOCK_SIZE: ',block_size)
    #for layer in pixels:
    for layer in pixels:
        print(layer)
        encoding, code_table = adaptive_encoding_stream( pixels[layer] , block_size )
        if layer in switching_activity:
            switching_activity[layer].append( get_sa_stream_avg(encoding) )
            mem_size[layer].append( sum([ len(i) for i in code_table ])/TEST_SIZE )
        else:
            switching_activity[layer] = [ get_sa_stream_avg(encoding) ]
            mem_size[layer] = [ sum([ len(i) for i in code_table ])/TEST_SIZE ]


fig, ax1 = plt.subplots()
for layer in pixels:
    ax1.plot(block_sizes,  switching_activity[layer], label=layer, linestyle='--')
ax1.set_xlabel('Block Size')
ax1.set_ylabel('Switching Activity (%)')

ax2 = ax1.twinx()
for layer in pixels:
    ax2.plot(block_sizes, mem_size[layer], label=layer, linestyle='-' )
ax2.set_ylabel('Memory Size')
plt.legend()
plt.grid()
plt.show()

'''
