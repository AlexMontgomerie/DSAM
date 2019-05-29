from sa import *
from operator import itemgetter
from joblib import Parallel, delayed
import dill
import multiprocessing
from multiprocessing import Pool


def int2bin(n):
    bits = []
    for i in range(FIXED_WIDTH):
        bits.append((n >> i) & 1)
    return bits

def bin2int(n):
    val = 0
    for i in range(FIXED_WIDTH):
        val |= n[i] << i
    return val

def bin2gray(bits):
	return bits[:1] + [i ^ ishift for i, ishift in zip(bits[:-1], bits[1:])]

def gray_encoding_stream(stream):
    # encoded stream
    gray = []
    # encode each value in stream
    for val in stream:
        gray.append(bin2int(bin2gray(int2bin(val))))
    return gray

def bus_invert_stream(stream):
    encoded = [stream[0]]
    for i in range(1,len(stream)):
        if hamming_distance(stream[i],encoded[i-1]) > 8:
            encoded.append(np.invert(stream[i]))
        else:
            encoded.append(stream[i])
    return encoded

def correlator(prev, diff):
    return prev ^ diff

def adaptive_encoding_static_stream(stream):
    # count symbols in stream
    symbol_count = {}
    for val in stream:
        if val in symbol_count:
            symbol_count[val] += 1
        else:
            symbol_count[val] = 1

    # create code table
    scheme = sorted(symbol_count.items(), key=itemgetter(1))
    code_table = {}
    for i in range(len(scheme)-1):
        code_table[scheme[-(i+1)][0]] = i
    code_table[scheme[0][0]] = len(scheme)-1

    # encode stream
    prev = stream[0]
    encoded = [prev]
    for i in range(1,len(stream)):
        val = correlator( prev, code_table[stream[i]] )
        prev = val
        encoded.append(val)
    return encoded, code_table

def adaptive_encoding_stream(stream, block_size):
    # number of blocks
    num_blocks = math.floor(len(stream)/block_size) # - 1
    symbol_count = [{}] * num_blocks
    code_table   = [{}] * num_blocks
    #prev = 0
    encoded = [0] * len(stream)

    # block encoding function
    for block_index in range(num_blocks):
        prev = 0
        # get symbol count
        for i in range( block_index*block_size, min((block_index+1)*block_size, len(stream)) ):
            if stream[i] in symbol_count:
                symbol_count[block_index][stream[i]] += 1
            else:
                symbol_count[block_index][stream[i]] = 1
        # create code table
        scheme = sorted(symbol_count[block_index].items(), key=itemgetter(1))
        for i in range(len(scheme)-1):
            code_table[block_index][scheme[-(i+1)][0]] = i
        code_table[block_index][scheme[0][0]] = len(scheme)-1

        # encode stream
        for i in range( block_index*block_size, min((block_index+1)*block_size, len(stream)) ):
            val = correlator( prev, code_table[block_index][stream[i]] )
            prev = val
            encoded[i] = val

    return encoded, code_table

# V1

def differential_encoding_stream(stream, distance=1):
    encoded = np.bitwise_xor( stream[distance:len(stream)], stream[0:(len(stream)-distance)] )
    encoded = np.concatenate( (stream[0:distance], encoded) )
    return encoded

def differential_encoding_stream_decode(stream, distance=1):
    decoded = []
    # buffer initial values
    for i in range(distance):
        decoded.append(stream[i])
    # encode the rest
    for i in range(distance,len(stream)):
        decoded.append((stream[i]^decoded[i-distance]))
    return decoded

# V2
def differential_encoding_stream_2(stream, distance=1):
    encoded = np.subtract( stream[distance:len(stream)], stream[0:(len(stream)-distance)] )
    encoded = np.concatenate( (stream[0:distance], encoded) )
    encoded = np.absolute(encoded)
    encoded = np.bitwise_and( encoded, 0xFFFF )
    encoded_out = [encoded[0]]
    for i in range(1,len(encoded)):
        encoded_out.append(encoded[i]^encoded_out[i-1])
    return encoded

def differential_encoding_stream_2_decode(stream, distance=1):
    decoded = [stream[0]]
    # decorrelate
    for i in range(1,len(stream)):
        decoded.append(stream[i]^stream[i-1])
    decoded_out = decoded[:distance]
    for i in range(distance,len(stream)):
        decoded_out.append(decoded[i] + decoded_out[i-distance])
    return decoded_out

def differential_encoding_pure_stream(stream):
    encoded = [stream[0]]
    #encoded = stream[:distance]
    for i in range(1,len(stream)):
        encoded.append(stream[i] - stream[i-1])
    return np.array(encoded)


if __name__ == '__main__':

    tmp = [ 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF,
            0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF,
            0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF,
            0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF ]

    tmp = [ 1, 3443, 436, 3, 4436, 543, 9, 4321, 435, 1]
    #tmp = [ 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    #tmp = [ 1.78978, 0.9990, 0.9999, 1.798, 0.68, 0.679, 1.2345, 0.6757, 0.098676, 1.456 ]

    #encoded = differential_encoding_stream(tmp,3)
    #encoded2 = differential_encoding_stream_decode(encoded, 3)
    print(tmp)
    encoded = differential_encoding_stream_2(tmp,3)
    encoded2 = differential_encoding_stream_2_decode(encoded, 3)
    print("switching activity           : \t {sa}".format(sa=get_sa_stream_avg(tmp)) )
    print("switching activity (encoded) : \t {sa}".format(sa=get_sa_stream_avg(encoded)) )
    print("switching activity (encoded) : \t {sa}".format(sa=get_sa_stream_avg(encoded2)) )
    print(tmp)
    print(encoded)
    print(encoded2)
    #print(differential_encoding_stream_decode(encoded))
    #print(differential_encoding_stream_decode(differential_encoding_stream_decode(encoded2)))
