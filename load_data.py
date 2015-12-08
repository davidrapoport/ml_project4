import os, pdb
from sklearn.cross_validation import train_test_split
from scipy.sparse import csr_matrix, vstack
import numpy as np


files = ["aid_1815_data", 	"aid_463213_data",   "aid_624504_data", 
"aid_1851_1a2_data", 	"aid_463215_data",   "aid_651739_data", 
"aid_1851_2c19_data", 	"aid_488912_data",   "aid_651744_data", 
"aid_1851_2c9_data", 	"aid_488915_data",   "aid_652065_data", 
"aid_1851_2d6_data", 	"aid_488917_data",   
"aid_1851_3a4_data", 	"aid_488918_data",   
"aid_1915_data", 	"aid_492992_data", "aid_2358_data", "aid_504607_data" ]

if not os.path.exists("data/test_inputs.npy"):
	#We need to generate the npy files and make a test/train split
	np.random.seed(12349)
	test_inputs = None
	test_tasks = None
	test_outputs = None
	total = 0
	for num,f in enumerate(files):
		with open("data/%s.csv"%f, "r") as csv:
			arr = np.loadtxt(csv, delimiter=',', skiprows=1)
			np.random.shuffle(arr)
			total += arr.size
			cids = arr[:,0]
			arr = arr[:,1:]
			targets = arr[:,-1]
			xs = arr[:,:-1]
			Xtrain, Xtest, Ytrain, Ytest, Cidtrain, Cidtest= train_test_split(xs, targets, cids, test_size=0.2, random_state=8753)
			np.save("data/task_%d_inputs"%(num+1), csr_matrix(Xtrain))
			np.save("data/task_%d_outputs"%(num+1), Ytrain)
			np.save("data/task_%d_cids"%(num+1), Cidtrain)
			if test_inputs is None:
				test_inputs = csr_matrix(Xtest)
				test_tasks = np.ones((Xtest.shape[0],1))
				test_outputs = Ytest
			else:
				# pdb.set_trace()
				test_inputs = vstack((test_inputs, csr_matrix(Xtest)))
				temp = np.ones((Xtest.shape[0],1))*num
				test_tasks = np.vstack((test_tasks, temp))
				test_outputs = np.concatenate((test_outputs, Ytest))
	np.save("data/test_inputs", test_inputs)
	np.save("data/test_outputs", test_outputs)
	np.save("data/test_tasks", test_tasks)



def get_minibatches(batch_size, add_bias=False):
	pass
