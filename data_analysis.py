import numpy as np 
from scipy.sparse import csr_matrix
import os, csv, json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt


if not os.path.exists("analysis.json"):
	info = {}
	for csv_file in os.listdir("./data"):
		d = {}
		if not csv_file.endswith(".csv"):
			continue
		with open("data/"+csv_file, "r") as f:
			arr = np.loadtxt(f, delimiter=',', skiprows=1)
			x = arr[:,1:-1]
			y = arr[:,-1]
			d['num_features'] = x.shape[1] 
			d['num_candidates'] = x.shape[0]
			# d['cids'] = [int(cid) for cid in x[:,0].tolist()]
			d['num_not_zeros'] = (x >0).sum()
			d['percent_not_zeros'] = float(d['num_not_zeros'])/ x.size
			d['num_activated'] = (y>0).sum()
			d['percent_activated']= float((y>0).sum())/y.size
			# if (arr == 21138.0).sum()>1:
			# 	print (arr == 21138.0).sum()
			# 	print csv_file
			info[csv_file] = d
			# print d
	num_tasks = len(info.keys())
	total_activated = 0.0
	total_nonzero = 0.0
	total_examples = 0.0
	for f in info.values():
		total_examples += f['num_candidates']
		total_nonzero += f['num_not_zeros']
		total_activated += f['num_activated']
	info['average_activated'] = float(total_activated)/float(total_examples)
	info['average_nonzero'] = float(total_nonzero)/float(total_examples*4096.0)
	info['total_examples'] = total_examples
	info['total_nonzero'] = total_nonzero
	info['total_activated'] = total_activated

	with open("analysis.json","w") as f:
		json.dump(info, f)
else:
	with open("analysis.json", "r") as f:
		info = json.load(f)

def get_intimate():
	cids = defaultdict(int)
	for f in info:
		for cid in info[f]['cids']:
			cids[cid]+= 1
	tups = cids.items()
	tups.sort(key=lambda x: -1*x[1])
	print tups[:10]
	freqs = Counter(cids.values())
	print freqs

def create_timeline():
	with open("timeline.json","r") as f:
		data = json.load(f)
	batches, omtl, kpercept, knb = zip(*data)
	plt.plot(batches[::20], omtl[::20], 'r--', batches[::20], kpercept[::20], 'b--', batches[::20], knb[::20], 'g^')
	plt.xlabel("Number of examples")
	plt.ylabel("Accuracy")
	plt.legend(['OMTL', 'KPerceptron', 'KNB'])
	# plt.show()
	plt.savefig("report/omtl_timeline.pdf", format="pdf")

def confusion_matrix():
	def plot_confusion_matrix(cm, title='Task Relatedness', cmap=plt.cm.Blues):
	    plt.imshow(cm, interpolation='nearest', cmap=cmap)
	    plt.title(title)
	    plt.colorbar()
	    tick_marks = np.arange(20)
	    plt.xticks(tick_marks, range(20), rotation=45)
	    plt.yticks(tick_marks, range(20))
	    plt.tight_layout()
	    plt.savefig("report/task_relate.pdf", format="pdf")


	# Compute confusion matrix
	with open("task_relate.json", "r") as f:
		data = np.asarray(json.load(f))
	plot_confusion_matrix(data)
	cm = data
	# Normalize the confusion matrix by row (i.e by the number of samples
	# in each class)
	cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	# plt.figure()
	# plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

	# plt.show()

# confusion_matrix()