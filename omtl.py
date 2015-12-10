import numpy as np 
from sklearn.linear_model import SGDClassifier
from itertools import izip
from util.load_data import get_minibatches, data
import pdb, time

class Perceptron(object):

	def __init__(self, feature_len):
		self.w = np.random.randn(1, feature_len)
		self.w_prev = np.random.randn(1, feature_len)


	def update_weight(self, alph, x, label):
		self.w_prev = self.w
		self.w = self.w + label*alph*x

	def predict(self, x):
		return np.sign(np.dot(self.w, x.transpose()))



class OMTL(object):

	def __init__(self, num_tasks=20, epoch=80, feature_len=4097, divergence="vonneumann", matrix_interaction_lr=0.001):
		self.num_tasks = num_tasks
		self.epoch = epoch
		self.t = 0
		self.alpha = matrix_interaction_lr
		if divergence not in ("logdet","vonneumann"):
			raise ValueError('Divergence must be one of ("logdet","vonneumann")')

		self.divergence = divergence
		self.A = np.eye(num_tasks) * (1. / num_tasks)
		self.Aprev = self.A
		self.W = None
		self.Wprev = None
		self.Ainv = np.linalg.inv(self.A)
		self.learners = []
		for i in range(num_tasks):
			self.learners.append(Perceptron(feature_len))




	def fit(self, tasks, X, y):
		# expects length of tasks X and y to be the same
		# returns an iterator 
		for task, inp, label in izip(tasks,X,y):
			self.t += 1
			output = self.learners[task].predict(inp)
			if output == label:
				continue
			for i in range(self.num_tasks):
				self.learners[i].update_weight(self.Ainv[task,i], inp, label)
			if self.t > self.epoch:
				self.Aprev = self.A
				W = []
				for i in range(self.num_tasks):
					W.append(self.learners[task].w_prev.ravel())
				W = np.asarray(W).transpose()
				self.Wprev = self.W
				self.W = W
				temp = np.dot(W.transpose(), W)
				if self.divergence == "logdet":
					self.A = np.linalg.pinv( self.Ainv + self.alpha*(temp+temp.transpose())/ 2.  )
				elif self.divergence == "vonneumann":
				 	self.A = np.exp( np.log(self.A) + self.alpha*(temp+temp.transpose())/ 2.  )
				self.Ainv = np.linalg.pinv(self.A)


	def predict(self, task_num, X):
		pred = self.learners[task_num].predict(X)
		return (pred>0).astype(int)

	def score(self, tasks, x, y):
		correct = 0.0
		for task, x, y in izip(tasks, x, y):
			pred = self.predict(int(task), x)
			if pred[0] == y:
				correct += 1.0
		return correct / float(x.shape[0])

	def get_norms(self):
		if self.W is None:
			return 0.0, 0.0
		return np.linalg.norm(self.A-self.Aprev), np.linalg.norm(self.W - self.Wprev)



def main():
	val_tasks, val_x, val_y = np.ndarray(0), None, np.ndarray(0)
	for task, inp, outp in get_minibatches(batch_size=200, num_epochs=1, add_bias=True):
		val_tasks = np.hstack((val_tasks, task.ravel()))
		val_y = np.hstack((val_y, outp.ravel()))
		if val_x is None:
			val_x = inp
		else:
			val_x = np.vstack((val_x, inp))
	learner = OMTL(epoch=20*5)
	start = time.time()
	for cnt, (task, inp, outp) in enumerate(get_minibatches(
								batch_size=1, num_epochs=10000, add_bias=True)):
		learner.fit(task[0].astype(int), inp, outp)
		if (cnt % 500) ==0:
			score = learner.score(val_tasks, val_x, val_y)
			anorm, wnorm = learner.get_norms()
			print "After %.2f s and %d it acc: %f , norm A'-A: %.2f , norm W'-W: %.2f " % ((time.time() - start), cnt, score, anorm, wnorm)



if __name__ == '__main__':
	main()