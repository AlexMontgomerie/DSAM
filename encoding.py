from sa import *
from operator import itemgetter

BLOCK_SIZE = 10

# TODO: find number of ones in a word

def adaptive_encoding(im, block_size):
    #fold input
    im_folded = []
    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            for channel in range(im.shape[2]):
                im_folded.append(int(im[row,col,channel]))

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

    size = 0
    for i in range(len(encoding)):
        size += len(encoding[i])
    #
    print("average sa: ",(sum(im_sa)/float(len(im_sa))), " , ",(sum(im_encoded_sa)/float(len(im_encoded_sa))) )

    print("size needed: ", size)

    #plt.show()

if __name__ == '__main__':
    # get image from filepath
    im = Image.open('data/alexnet.jpg')
    # save as numpy array
    im = np.array(im,dtype=np.float32)

    block_sizes = [10,100,200,500,1000,2000,5000,10000]
    for block_size in block_sizes:
        adaptive_encoding(im,block_size)
