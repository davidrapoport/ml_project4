import csv
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np

def output_sorted_activities(activity_file, label_index, headerskip):
    print activity_file
    outputs = []
    with open(activity_file) as actfile:
        csvreader = csv.reader(actfile, delimiter=',')
        for i in xrange(headerskip):
            csvreader.next()
        for line in csvreader:
            try:
                pair = (int(line[2]), line[label_index])
            except ValueError, e:
                continue
            outputs.append(pair)
    outputs.sort(key=lambda tup: tup[0])
    return outputs


def output_smile_hash(compounds_file):
    toret = {}
    with open(compounds_file, 'r') as infile:
        for line in infile:
            splitted = line.split('\t')
            cid = splitted[0]
            smile = splitted[1].rstrip()
            toret[cid] = smile
    return toret


def write_csv(compounds_file, activity_file, activity_index, headerskip, name):
    # instantiate the vars we're going to return
    rows = []
    Y = []

    # get the list of activities that we actually want smiles for
    # and the smiles
    activities = output_sorted_activities(
        activity_file, activity_index, headerskip)
    smile_hash = output_smile_hash(compounds_file)

    # go through the list of activities and do a bunch of stuff
    # the mol object will have 'id' in the first entry, 'activity' in the
    # second
    for idx, mol in enumerate(activities):
        if mol[1] == 'Inconclusive':
            pass
            # print('deleting a row')
        elif (str(mol[0]) not in smile_hash):
            print('can\'t find this molecule yo!')
        else:
            if (mol[1] == 'Active'):
                activity = 1
            else:
                activity = 0
            # get the molecule
            molsmile = smile_hash[str(mol[0])]
            mole = Chem.MolFromSmiles(molsmile)
            vect = AllChem.GetMorganFingerprintAsBitVect(mole, 3, nBits=4096)
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(vect, arr)
            arr = np.insert(arr, 0, mol[0])

            # append our vect to arr
            rows.append(arr)
            # append the target to Y
            Y.append(activity)
    X = np.array(rows)
    Ynew = np.array(Y)
    assert X.shape[0] == Ynew.shape[0]
    numfeats = X.shape[1] - 1
    with open('./data/' + name + '.csv', 'wb') as writefile:
        writer = csv.writer(writefile, delimiter=',')
        writer.writerow(['id'] + ['Feature'] * numfeats + ['Label'])
        # print("numfeats %d" % numfeats)
        # print(X.shape, Ynew.shape)
        X = np.hstack((X, Ynew.reshape(Ynew.shape[0], 1)))
        for idx in xrange(X.shape[0]):
            writer.writerow(X[idx])

if __name__ == '__main__':
    idlist = [1915, 1815]
    actidx = 3
    headerskip = 6
    for filenum in idlist:
        compfile = "./data/compounds/aid_%d_compounds.txt" % filenum
        actfile = "./data/assays/aid_%d_assays.csv" % filenum
        name = "aid_%d_data" % filenum
        write_csv(compfile, actfile, actidx, headerskip, name)
    idlist = [2358, 463213, 463215, 488912, 488915, 488917, 488918, 492992, 504607, 624504, 651739, 651744, 652065]
    actidx = 3
    headerskip = 5
    for filenum in idlist:
        compfile = "./data/compounds/aid_%d_compounds.txt" % filenum
        actfile = "./data/assays/aid_%d_assays.csv" % filenum
        name = "aid_%d_data" % filenum
        write_csv(compfile, actfile, actidx, headerskip, name)
    filenum = 1851
    actidxs = [11, 38, 65, 92, 119]
    namelist = ['2c19', '2d6', '3a4', '1a2', '2c9']
    headerskip = 9
    compfile = './dta/compounds/aid_%d_compounds.txt' % filenum
    actfile = "./data/assays/aid_%d_assays.csv" % filename
    for idx, act in enumerate(actidxs):
        name = "aid_1851_%s_data" %  namelist[idx]
        write_csv(compfile, actfile, act, headerskip, name)
