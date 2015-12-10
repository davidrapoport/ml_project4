
import numpy as np 
import gnumpy as gnp
import itertools
import gdnn as dnn
import os
import urllib
import subprocess
import h5py

import gzip
import struct
from array import array

#set to an appropriate value for your GPU, if you are using one
dnn.gnp.max_memory_usage = 2800000000

def loadData():
    return [], [] , [] , []

def sampleMinibatch(batch_size, inputs, targets):
    i = np.random.randn(batch_size, 1000)
    return i, (i.mean(1)>0)

def main():
    epochs = 1
    mbsz = 64
    num_tasks = 20
    mbPerEpoch = int(num.ceil(60000./mbsz))
    layerSizes = [784, 512, 512]
    scales = [0.05]*(len(layerSizes)-1)
    weightCosts = [0] * len(scales)
    learnRate = 0.1

    trainInps, trainTargs, testInps, testTargs = loadData()
    num.random.seed(5)
    mbStream = (sampleMinibatch(mbsz, trainInps, trainTargs) for unused in itertools.repeat(None))

    
    inpLay0 = dnn.InputLayer('inp0', layerSizes[0])
    hidLay0 = dnn.Sigmoid('hid0', layerSizes[1])
    hidLay1 = dnn.Sigmoid('hid1', layerSizes[2])

    
    # outLay0 = dnn.Softmax('out0', layerSizes[-1], k = layerSizes[-1])
    
    layers = [inpLay0, hidLay0, hidLay1, outLay0]
    edges = []
    for i in range(1, len(layers)):
        W = gnp.garray(scales[i-1]*num.random.randn(layerSizes[i-1],layerSizes[i]))
        bias = gnp.garray(num.zeros((1,layerSizes[i])))
        edge = dnn.Link(layers[i-1], layers[i], W, bias, learnRate, momentum = 0.9, L2Cost = weightCosts[i-1])
        edges.append(edge)

    net = dnn.DAGDNN(layers, edges)

    valCE, valErr = getCEAndErr(net, testInps, testTargs)
    print 'valCE = %f, valErr = %f' % (valCE, valErr)
    for ep, (CEs, errs) in enumerate(net.train(mbStream, epochs, mbPerEpoch, lossFuncs = [numMistakesLoss])):
        valCE, valErr = getCEAndErr(net, testInps, testTargs)
        print ep, 'trCE = %f, trErr = %f' % (CEs['out0'], errs)
        print 'valCE = %f, valErr = %f' % (valCE, valErr)

    with h5py.File('mnistNet.hdf5', mode='w', driver = None, libver='latest') as fout:
        net.save(fout)

if __name__ == "__main__":
    main()