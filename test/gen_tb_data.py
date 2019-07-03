from src.encoding import *
import random
import json

TEST_SIZE   = 10
FIXED_WIDTH = 16 
CHANNELS    = 4

# generate random data stream
data_in = [ random.randint(0,2**FIXED_WIDTH-1) for i in range(TEST_SIZE) ]

# run coding scheme
correct, _ =  dsam_encoding_stream(data_in, CHANNELS)

info = {
      'data_in' : data_in,
      'correct' : correct,
      'size'       : TEST_SIZE,
      'data_width' : FIXED_WIDTH,
      'channels'   : CHANNELS,
      'test_num'   : 1
}
print(info)
# save to json
with open('data/data.json', 'w') as f:  
    json.dump(info, f)

