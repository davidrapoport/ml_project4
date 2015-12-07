import numpy as np 
from scipy.sparse import csr_matrix
import os, csv, json
from collections import Counter, defaultdict


if not os.path.exists("analysis.json"):
	info = {}
	for csv_file in os.listdir("./data"):
		d = {}
		if not csv_file.endswith(".csv"):
			continue
		with open("data/"+csv_file, "r") as f:
			arr = np.loadtxt(f, delimiter=',', skiprows=1)
			d['num_features'] = arr.shape[1]
			d['num_candidates'] = arr.shape[0]
			d['cids'] = [int(cid) for cid in arr[:,0].tolist()]
			d['num_not_zeros'] = (arr >0).sum()
			d['percent_not_zeros'] = float(d['num_not_zeros'])/ arr.size
			# if (arr == 21138.0).sum()>1:
			# 	print (arr == 21138.0).sum()
			# 	print csv_file
			info[csv_file] = d
			# print d
	with open("analysis.json","w") as f:
		json.dump(info, f)
else:
	with open("analysis.json", "r") as f:
		info = json.load(f)

cids = defaultdict(int)
for f in info:
	for cid in info[f]['cids']:
		cids[cid]+= 1
tups = cids.items()
tups.sort(key=lambda x: -1*x[1])
print tups[:10]
freqs = Counter(cids.values())
print freqs