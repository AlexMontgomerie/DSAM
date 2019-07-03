from src.sa import *
from operator import itemgetter

# helper functions
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

# GRAY ENCODING

def bin2gray(bits):
	return bits[:1] + [i ^ ishift for i, ishift in zip(bits[:-1], bits[1:])]

def gray_encoding_stream(stream):
    # encoded stream
    gray = []
    # encode each value in stream
    for val in stream:
        gray.append(bin2int(bin2gray(int2bin(val))))
    return gray

# BUS-INVERT ENCODING

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

# CSAM

def csam_encoding_stream(stream, distance=1):
    encoded = np.bitwise_xor( stream[distance:len(stream)], stream[0:(len(stream)-distance)] )
    encoded = np.concatenate( (stream[0:distance], encoded) )
    return encoded

def csam_decoding_stream(stream, distance=1):
    decoded = []
    # buffer initial values
    for i in range(distance):
        decoded.append(stream[i])
    # encode the rest
    for i in range(distance,len(stream)):
        decoded.append((stream[i]^decoded[i-distance]))
    return decoded

# DSAM
def dsam_encoding_stream(stream, distance=1):
    encoded = np.subtract( stream[distance:len(stream)], stream[0:(len(stream)-distance)] )
    encoded = np.concatenate( (stream[0:distance], encoded) )
    sign    = np.sign(encoded)
    encoded = np.absolute(encoded)
    #encoded = np.bitwise_and( encoded, 0xFFFF )
    encoded_out = [encoded[0]]
    for i in range(1,len(encoded)):
        encoded_out.append(encoded[i]^encoded_out[i-1])
    return encoded_out,sign

def dsam_decoding_stream(stream,sign,distance=1):
    decoded = [stream[0]]
    # decorrelate
    for i in range(1,len(stream)):
        decoded.append(stream[i]^stream[i-1])
    print(decoded)
    decoded_out = decoded[:distance]
    for i in range(distance,len(stream)):
        decoded_out.append((decoded[i]*sign[i] + decoded_out[i-distance]))
    return decoded_out

if __name__ == '__main__':

    data_in = [ 1, 2, 3, 4, 5, 6, 7, 8 ]
    print(dsam_encoding_stream(data_in,4))
