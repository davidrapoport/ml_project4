import numpy as np 
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from itertools import izip
from load_data import get_minibatches, data
import pdb, time, json

err = open("omtl_error.log", "a")

class Perceptron(object):
	## Single Perceptron, predicts SIGN(w*x)
	## Performs updates by w' = w + y*alph*x for incorrectly predicted labels
	def __init__(self, feature_len):
		self.w = np.random.randn(1, feature_len)
		self.w_prev = np.random.randn(1, feature_len)


	def update_weight(self, alph, x, label):
		# Perceptron expects the two labels to be -1,1 so we convert 0 to -1
		if label==0:
			label = -1.0
		# Save the previous W for use in OMTL
		self.w_prev = self.w
		# Perform the update
		self.w = self.w + label*alph*x

	def predict(self, x):
		# returns SIGN(w*x) which will be -1,0,1
		return np.sign(np.dot(self.w, x.transpose()))

class KPerceptrons(object):
	# K independent perceptrons, no shared learning occurs
	# The tunable hyper parameter is the learning_rate. We found
	# that for our dataset 10,000 was optimal

	def __init__(self, num_tasks=20, learn_rate=1, feature_len=4097):
		self.num_tasks = num_tasks
		self.learn_rate = learn_rate
		self.feature_len = feature_len
		self.learners = []
		self.t = 0
		for i in range(num_tasks):
			self.learners.append(Perceptron(feature_len))

	def fit(self, tasks, X, y):
		# expects length of tasks X and y to be the same
		# For each triple (task, input, label) we predict and update
		# With perceptron #task
		for task, inp, label in izip(tasks,X,y):
			self.t += 1
			output = self.learners[task].predict(inp)
			if output == label:
				continue
			self.learners[task].update_weight(self.learn_rate, inp, label)
			

	def predict(self, task_num, X):
		pred = self.learners[task_num].predict(X)
		return (pred>0).astype(int)

	def score(self, tasks, inps, outps):
		correct = 0.0
		for task, x, y in izip(tasks, inps, outps):
			pred = self.predict(int(task), x)
			if pred[0] == y:
				correct += 1.0
		return correct / float(inps.shape[0])


class KNaiveBayes():
	# K independent Multinomial Bayes to compare with OMTL and K Independent Perceptrons
	def __init__(self, num_tasks=20):
		self.num_tasks = num_tasks
		self.learners = []
		for i in range(num_tasks):
			self.learners.append(MultinomialNB())

	def fit(self, tasks, X, y):
		# Join tasks, X, and y into one matrix
		complete = np.hstack((tasks,X,y))
		for i in range(self.num_tasks):
			# Get the elements which correspond to task i
			task = complete[complete[:,0]==i]
			if not task.size:
				continue
			# Seperate the data again
			t = task[:,0]
			label = task[:,-1]
			x = task[:,1:-1]
			# Partial fit, last parameter is an array of all possible labels

			self.learners[i].partial_fit(x,label,np.array([0,1]))

	def score(self, tasks, inps, labels):
		# Join tasks, inps, and labels into one matrix
		# Reshape calls are made to convert from shape=(num_examples,) 
		# to shape=(num_examples,1) so that we can call hstack
		complete = np.hstack((tasks.reshape((-1,1)),inps,labels.reshape((-1,1)) ))
		total = 0.0
		for i in range(self.num_tasks):
			task = complete[complete[:,0]==i]
			t = task[:,0]
			label = task[:,-1]
			x = task[:,1:-1]
			score = self.learners[i].score(x,label)
			# multiply score by the number of examples 
			# for the task to get how many were correctly classified
			total += score*x.shape[0]
		return total / float(inps.shape[0])


class OMTL(object):

	def __init__(self, num_tasks=20, epoch=80, feature_len=4097, divergence="logdet", matrix_interaction_lr=0.001):
		self.num_tasks = num_tasks
		# Sometimes A matrix gets too large, if this happens we don't want to 
		# continue doing computations
		self.has_diverged=False
		# Which epoch do we begin changing A at
		self.epoch = epoch
		# Keeps track of the current epoch
		self.t = 0
		# Learning rate for the matrix interaction matrix A
		self.alpha = matrix_interaction_lr
		if divergence not in ("logdet","vonneumann"):
			raise ValueError('Divergence must be one of ("logdet","vonneumann")')

		self.divergence = divergence
		# Initialize matrix interaction matrix to the identity multiplied by
		# 1/num_tasks. 
		self.A = np.eye(num_tasks) * (1. / num_tasks)
		self.Aprev = self.A
		self.W = None
		self.Wprev = None
		self.Ainv = np.linalg.inv(self.A)
		# We will also initialize num_tasks seperate perceptrons
		self.learners = []
		for i in range(num_tasks):
			self.learners.append(Perceptron(feature_len))




	def fit(self, tasks, X, y):
		# expects length of tasks X and y to be the same
		if self.has_diverged:
			return
		for task, inp, label in izip(tasks,X,y):
			self.t += 1
			output = self.learners[task].predict(inp)
			# If we predicted correctly we don't do updates
			if output == label:
				continue
			# Update all of the tasks i using the learning rate in A[task, i]
			for i in range(self.num_tasks):
				self.learners[i].update_weight(self.Ainv[task,i], inp, label)
			# If we are ready to begin updating A
			if self.t > self.epoch:
				self.Aprev = self.A
				# Create W an K*d matrix where K is the number of tasks
				# and d is the size of the feature vector
				W = []
				for i in range(self.num_tasks):
					W.append(self.learners[task].w_prev.ravel())
				W = np.asarray(W).transpose()
				self.Wprev = self.W
				self.W = W
				temp = np.dot(W.transpose(), W)
				if self.divergence == "logdet":
					try:
						self.A = np.linalg.pinv( self.Ainv + self.alpha*(temp+temp.transpose())/ 2.  )
					except Exception, e:
						self.has_diverged = True
						err.write("OMTL with lr %f epochs %f diverged after %d iterations" % (self.alpha, self.epoch, self.t) )
				elif self.divergence == "vonneumann":
				 	self.A = np.exp( np.log(self.A) + self.alpha*(temp+temp.transpose())/ 2.  )
				self.Ainv = np.linalg.pinv(self.A)


	def predict(self, task_num, X):
		pred = self.learners[task_num].predict(X)
		return (pred>0).astype(int)

	def score(self, tasks, inps, labels):
		correct = 0.0
		for task, x, y in izip(tasks, inps, labels):
			pred = self.predict(int(task), x)
			if pred[0] == y:
				correct += 1.0
		return correct / float(inps.shape[0])

	def get_norms(self):
		# returns |A-Aprev|, |W - Wprev|. Used to debug to make sure the weight vectors
		# and interaction matrix is updating
		if self.W is None or self.Aprev is None or self.Wprev is None:
			return 0.0, 0.0
		return np.linalg.norm(self.A-self.Aprev), np.linalg.norm(self.W - self.Wprev)



def get_validation_set(batch_size=1500):
	# Returns 3 arrays (tasks, X, y) with batch_size*20 examples in each 
	val_tasks, val_x, val_y = np.ndarray(0), None, np.ndarray(0)
	for task, inp, outp in get_minibatches(batch_size=batch_size, num_epochs=1, add_bias=True):
		val_tasks = np.hstack((val_tasks, task.ravel()))
		val_y = np.hstack((val_y, outp.ravel()))
		if val_x is None:
			val_x = inp
		else:
			val_x = np.vstack((val_x, inp))
	return val_tasks, val_x, val_y

def grid_search():
	out = open("omtl_log", "w")
	val_tasks, val_x, val_y = get_validation_set()
	omtl_lrs = np.logspace(-28,-14,8)
	perceptron_lrs = np.logspace(-10,4,8)
	epochs = [20, 40, 80, 160, 320, 640, 1280]
	start = time.time()
	kpercept_results = []
	omtl_results = []
	for cnt, (omtl_lr, perceptron_lr) in enumerate(zip(omtl_lrs, perceptron_lrs)):
		omtls = [OMTL(epoch=epoch, matrix_interaction_lr=omtl_lr, divergence="logdet") for epoch in epochs]
		kpercepts = KPerceptrons(learn_rate=perceptron_lr)
		for batch, (task, inp, outp) in enumerate(get_minibatches(
									batch_size=1, num_epochs=5000, add_bias=True)):
			[learner.fit(task[0].astype(int), inp, outp) for learner in omtls]
			kpercepts.fit(task[0].astype(int), inp, outp)
			if batch%500 ==0:
				print "Batch %d" % batch
		kpercept_results.append((perceptron_lr, kpercepts.score(val_tasks, val_x, val_y)))
		for omtl in omtls:
			if omtl.has_diverged:
				omtl_results.append((omtl_lr, omtl.epoch, 0.0))
			else:
				omtl_results.append((omtl_lr, omtl.epoch, omtl.score(val_tasks, val_x, val_y)))
		print "Done with iteration %d of %d" %(cnt, len(omtl_lrs))
	json.dump({"omtl":omtl_results, "kpercepts":kpercept_results}, out)
	out.close()
		# if (cnt % 500) ==0:
		# 	score = learner.score(val_tasks, val_x, val_y)
		# 	base_score = base.score(val_tasks, val_x, val_y)
		# 	anorm, wnorm = learner.get_norms()
		# 	print "After %.2f s and %d it acc: %f , norm A'-A: %.2f , norm W'-W: %.2f " % ((time.time() - start), cnt, score, anorm, wnorm)
		# 	print "Base learner scored %f" % base_score

def timeline():
	# Generates the data file to be used to create the examples vs accuracy graph
	val_tasks, val_x, val_y = get_validation_set()
	learner = OMTL(epoch=40, matrix_interaction_lr=1e-20, divergence="logdet")
	kpercepts = KPerceptrons(learn_rate=10000)
	knb = KNaiveBayes()
	results = []
	tasks, x, y = [], None, []

	for batch, (task, inp, outp) in enumerate(get_minibatches(
								batch_size=1, num_epochs=8000, add_bias=True)):
		learner.fit(task[0].astype(int), inp, outp)
		kpercepts.fit(task[0].astype(int), inp, outp)

		# Used so we only call KNB.fit once per 100 examples
		tasks.append(int(task[0][0]))
		y.append(outp[0].astype(int))
		if x is None:
			x = inp
		else:
			x = np.vstack((x,inp))

		if batch%100 ==0 and batch>50:
			knb.fit(np.asarray(tasks).reshape((x.shape[0],1)), x, np.asarray(y).reshape((x.shape[0],1)))

			omtl_score = learner.score(val_tasks, val_x, val_y)
			kpercept_score = kpercepts.score(val_tasks, val_x, val_y)
			knb_score = knb.score(val_tasks, val_x, val_y)
			print "Batch %d" % batch
			print "OMTL score: %f" % omtl_score
			print "KPercept score %f" % kpercept_score
			print "KNB score %f" % knb_score
			results.append((batch, omtl_score, kpercept_score, knb_score))

			tasks, x, y = [], None, []

	with open("timeline.json", "w") as f:
		json.dump(results, f)


def test():
	learner = OMTL(epoch=640, matrix_interaction_lr=1e-10, divergence="logdet")
	kpercepts = KPerceptrons(learn_rate=10000)
	knb = KNaiveBayes()
	# Will spit out 20 batches of size 8000
	for task, inp, outp in get_minibatches(batch_size=8000, num_epochs=1, add_bias=True):
		knb.fit(np.asarray(task).reshape((inp.shape[0],1)), inp, np.asarray(outp).reshape((inp.shape[0],1)))
	# Will spit out 20*3000 batches of size 1
	for task, inp, outp in get_minibatches(batch_size=1, num_epochs=3000, add_bias=True):
		learner.fit(task[0].astype(int), inp, outp)
		kpercepts.fit(task[0].astype(int), inp, outp)
	# Test_tasks.shape = (28143, 1)
	test_tasks = np.load("data/test_tasks.npy")
	# We need to call item on test_inputs because it is an array that contains a sparse matrix
	# We then gotta convert it from sparse scipymatrix to nparray and add a bias 
	test_inputs = np.load("data/test_inputs.npy").item().toarray()
	test_inputs = np.hstack((np.ones((test_inputs.shape[0],1)), test_inputs))
	# Test_outputs.shape = (28143,)
	# So we need to reshape it to a (28143, 1) array
	test_outputs = np.load("data/test_outputs.npy").reshape((test_tasks.shape[0],1))
	# See KNB.score to see how I manipulate these three arrays so that they fit with normal
	# SKlearn learners
	print "KNB: %f" %knb.score(test_tasks, test_inputs, test_outputs)
	print "OMTL: %f" % learner.score(test_tasks, test_inputs, test_outputs)
	print "kpercepts: %f" % kpercepts.score(test_tasks, test_inputs, test_outputs)
	print json.dumps(learner.A.tolist())
	with open("task_relate.json","w") as f:
		json.dump(learner.A.tolist(), f)





if __name__ == '__main__':
	test()
	err.close()
