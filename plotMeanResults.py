import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import gzip
import cPickle
import sys

# Do this graph/results, add bad graph to paper and discuss failure of mtl, discussions

def plot(mean_train_error_array, val_error_array):
        training, = plt.plot(range(len(mean_train_error_array)), mean_train_error_array, 'p--', label='Training')
        validation, = plt.plot(range(len(val_error_array)), val_error_array, 'g--', label='Validation')
        plt.ylabel('Error')
        plt.xlabel('Epoch')
        plt.legend(handler_map={training: HandlerLine2D(numpoints=2), validation: HandlerLine2D(numpoints=2)})
        plt.show()

if __name__ == '__main__':
	trainLengths[]
	valLengths[]
	for i in range(20):
		t = gzip.open("data/train_%d.pickle.gz" % i)
		v = gzip.open("data/validate_%d.pickle.gz" % i)
		tx, ty = cPickle.load(t)
		vx, vy = cPickle.load(v)
		trainLengths.append(len(tx))
		valLengths.append(len(vx))
	
