import os, pdb
import cPickle
import numpy as np
import gzip

files = ["aid_1815_data",   "aid_463213_data",   "aid_624504_data", 
"aid_1851_1a2_data",    "aid_463215_data",   "aid_651739_data", 
"aid_1851_2c19_data",   "aid_488912_data",   "aid_651744_data", 
"aid_1851_2c9_data",    "aid_488915_data",   "aid_652065_data", 
"aid_1851_2d6_data",    "aid_488917_data",   
"aid_1851_3a4_data",    "aid_488918_data",   
"aid_1915_data",    "aid_492992_data", "aid_2358_data", "aid_504607_data" ]

data = []
for i in range(len(files)):
    s = "../data/task_"+ str(i+1)
    inp = np.load(s+"_inputs.npy").item().toarray().astype(np.float32)
    outp = np.load(s+"_outputs.npy").astype(np.float32)
    trainX = inp[:len(inp)*9.0/10.0]
    trainY = outp[:len(inp)*9.0/10.0]
    valX = inp[len(inp)*9.0/10.0:]
    valY = outp[len(inp)*9.0/10.0:]
    cPickle.dump((trainX, trainY), gzip.open('data/train_'+ str(i+1) +'.pickle.gz','wb'), cPickle.HIGHEST_PROTOCOL)
    cPickle.dump((valX, valY), gzip.open('data/validate_'+ str(i+1) +'.pickle.gz','wb'), cPickle.HIGHEST_PROTOCOL)
