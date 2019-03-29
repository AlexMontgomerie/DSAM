from sa import *
from operator import itemgetter

def int2bin(n):
    bits = []
    for i in range(FIXED_WIDTH):
        bits.append((n >> i) & 1)

    return bits

def bin2gray(bits):
	return bits[:1] + [i ^ ishift for i, ishift in zip(bits[:-1], bits[1:])]

BLOCK_SIZE = 10

def gray_encoding(im):
    im_folded = []
    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            for channel in range(im.shape[2]):
                im_folded.append(int(im[row,col,channel]))

    im_folded = np.array(im_folded)

    gray = []

    for i in range(im_folded.shape[0]):
        #gray.append(bin2gray(int2bin(im_folded[i])))
        val = bin2gray(int2bin(im_folded[i]))
        if len(val) != FIXED_WIDTH:
            print("error")
        gray.append(val)

    sa = 0
    for i in range(1,len(gray)):
        for j in range(len(gray[i])):
            if gray[i-1][j] != gray[i][j]:
                sa += 1

    print(sa/len(gray))
# TODO: find number of ones in a word

def adaptive_encoding(im, block_size):
    #fold input
    im_folded = []
    for row in range(im.shape[3]):
        for col in range(im.shape[2]):
            for channel in range(im.shape[1]):
                im_folded.append(ap_fixed(im[0][channel][row][col],FIXED_WIDTH,FIXED_INT_SIZE))

    im_folded = np.array(im_folded)

    num_blocks = math.floor(im_folded.shape[0]/block_size) - 1
    im_folded = im_folded[0:(block_size*num_blocks)]

    im_encoded = []
    encoding = []

    for i in range(num_blocks):
    #for i in range(1):
        scheme = {}
        #scheme = [0 for x in range(2**8)]
        for j in range(block_size):
        #    scheme[im_folded[i*BLOCK_SIZE+j]] += 1
            num = im_folded[i*block_size+j]
            if num in scheme:
                scheme[num] += 1
            else:
                scheme[num] = 1
        #get order of sorted list
        scheme = sorted(scheme.items(), key=itemgetter(1))
        code_word = []
        for j in range(len(scheme)-1):
            code_word.append(scheme[-(j+1)][0])
        code_word.append(scheme[0][0])

        for j in range(block_size):
        #    scheme[im_folded[i*BLOCK_SIZE+j]] += 1
            num = im_folded[i*block_size+j]
            for k in range(len(code_word)):
                if num == code_word[k]:
                    im_encoded.append(k)

        encoding.append(code_word)

    im_encoded = np.array(im_encoded)
    im_sa = [hamming_distance(im_folded[i],im_folded[i-1]) for i in range(1,im_folded.shape[0])]
    im_encoded_sa = [num_ones_in_word(im_encoded[i]) for i in range(im_encoded.shape[0])]

    plt.plot(im_encoded_sa[0:100])
    plt.plot(im_sa[0:100])

    plt.show()

    size = 0
    for i in range(len(encoding)):
        size += len(encoding[i])
    #
    print("average sa: ",(sum(im_sa)/float(len(im_sa)+1)), " , ",(sum(im_encoded_sa)/float(len(im_encoded_sa)+1)) )

    print("size needed: ", size)

    #plt.show()



if __name__ == '__main__':

    model_path    = 'model/alexnet.prototxt'
    data_path     = 'data/alexnet.jpg'
    weights_path  = 'weight/alexnet.caffemodel'

    # Initialise Network
    net = caffe.Classifier(model_path,weights_path)

    # Run for given data
    run_net(net,data_path)

    #block_size = 100

    block_size = 5000

    for layer in net.blobs:
        if layer == 'norm1' or layer == 'norm2':
            continue
        if layer == 'fc6':
            break
        print(layer)
        #gray_encoding(copy.deepcopy(net.blobs[layer].data[...]))
        adaptive_encoding(copy.deepcopy(net.blobs[layer].data[...]),block_size)

    block_sizes = [10,100,200,500,1000,2000,5000,10000]


    #block_sizes = [10]
    for block_size in block_sizes:
        adaptive_encoding(im,block_size)
