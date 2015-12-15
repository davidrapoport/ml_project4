import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import gzip
import cPickle
import sys

# Do this graph/results, add bad graph to paper and discuss failure of mtl, discussions

test_errs = [24.7524752475,36.1788617886,13.0952380952,20.5438066465,45.474137931,19.7026022305,21.5189873418,37.9032258065,19.926199262,19.4875776398,35.0,26.5258215962,16.4748201439,28.3870967742,20.0,28.5483870968,30.2469135802,24.593495935,30.2564102564,27.04]

train_items = [802, 5900, 4032, 10590, 3708, 4301, 10740, 4956, 4332, 10304, 4956, 3405, 11116, 4956, 10396, 4956, 2589, 3933, 1555, 4993]
test_items = [100, 737, 504, 1324, 464, 538, 1343, 620, 542, 1288, 620, 426, 1389, 620, 1300, 620, 324, 492, 194, 624]
val_items = test_items

def plot(mean_train_error_array, val_error_array):
        training, = plt.plot(range(len(mean_train_error_array)), mean_train_error_array, 'p--', label='Training')
        validation, = plt.plot(range(len(val_error_array)), val_error_array, 'g--', label='Validation')
        plt.ylabel('Error')
        plt.xlabel('Epoch')
        plt.legend(handler_map={training: HandlerLine2D(numpoints=2), validation: HandlerLine2D(numpoints=2)})
        plt.show()

if __name__ == '__main__':
	# trainLengths = []
	# valLengths = [] # The same as test
	# for i in range(20):
	# 	t = gzip.open("data/train_%d.pickle.gz" % i)
	# 	v = gzip.open("data/validate_%d.pickle.gz" % i)
	# 	tx, ty = cPickle.load(t)
	# 	vx, vy = cPickle.load(v)
	# 	trainLengths.append(len(tx))
	# 	valLengths.append(len(vx))

	test_correctly_classified = 0.0
	for i in range(20):
		test_correctly_classified = test_errs[i] * val_items[i]
	test_mean_accuracy = test_correctly_classified / sum(val_items)

	task_train_err = []
	task_val_err = []
	for i in range(20):
		train_log = open("dnn_%d.log" % (i+1))
		train_err = []
		val_err = []
		for line in train_log.readlines():
			if "training" in line:
				train_err.append(float(line.split(' ')[-2]))
			elif "validation" in line:
				val_err.append(float(line.split(' ')[-2]))
		plot(train_err, val_err)
		task_train_err.append(train_err)
		task_val_err.append(val_err)

	mean_train_err = np.zeros(50)
	mean_val_err = np.zeros(50)
	for j in range(50):
		for i in range(20):
			mean_train_err[j] += task_train_err[i][j]
			mean_val_err[j] += task_val_err[i][j]
		mean_train_err[j] = mean_train_err[j] / 20.0
		mean_val_err[j] = mean_val_err[j] / 20.0

	plot(mean_train_err, mean_val_err)

