from sa import *
from multiprocessing import Pool
import random

TEST_SIZE=100

model_path     = 'model/alexnet.prototxt'
data_path_root = '../fpgaConvNetSim/datasets/101_ObjectCategories'
weights_path   = 'weight/alexnet.caffemodel'

# Initialise Network
net = caffe.Classifier(model_path,weights_path)

data_files = []
index=0
for (dirpath, dirnames, filenames) in os.walk(data_path_root):
    for filename in filenames:
        data_files.append(dirpath+'/'+filename)

print("DATASET SIZE: ",len(data_files))

for f in data_files:
    # Run for given data
    run_net(net,f)
    # Analyse Network
    analyse_net(net)

'''
def sa_image(f):
    # Run for given data
    run_net(net,f)
    # Analyse Network
    analyse_net(net)

random_data_files = []

for i in range(TEST_SIZE):
    random_data_files.append( random.choice(data_files) )

pool = Pool()                          # Create a multiprocessing Pool
pool.map(sa_image, random_data_files)  # process data_inputs iterable with pool
'''
