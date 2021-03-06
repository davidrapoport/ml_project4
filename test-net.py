import numpy as np
import math
import sys
import datetime
from theano.tensor.shared_randomstreams import RandomStreams
from network.model.dropout_dnn import DNNDropout

shareLayers = True

num_tasks = 2
shared_layers_sizes = [512, 512]
task_specific_sizes = [[512, 512]] * 2

train_learning_rate = 150.0

input_size = 2
output_size = 2

mbatch_size = 1
num_bootstrap_rds = 500

valid_size = 1500
valid_mbatch_per_bootstrap = 600 / mbatch_size

bootstrap_size = 1500
mbatch_per_bootstrap = bootstrap_size / mbatch_size

shared_layers_num = len(shared_layers_sizes)

def log(string):

    sys.stderr.write(
        '[' + str(datetime.datetime.now()) + '] ' + str(string) + '\n')

if __name__ == '__main__':

    # we keep track of the training fns, valid fns, and networks
    train_fn_array = []
    valid_fn_array = []
    dnn_array = []

    np_rng = np.random.RandomState(848573)
    theano_rng = RandomStreams(np_rng.randint(2 ** 31))

    for n in xrange(num_tasks):
        log('> ... building the model for task %d' % (n))

        # set up the model
        dnn_shared = None
        shared_layers = []
        hidden_layers = shared_layers_sizes + task_specific_sizes[n]
        # use the first networks shared layers dawg
        if n > 0:
            dnn_shared = dnn_array[0]
            shared_layers = [m for m in xrange(shared_layers_num)]

        # create the network for the task
        # you can change the input dropout factor and the general dropout factor
        # look at the DNNDropout class
        if shareLayers:
            dnn = DNNDropout(np_rng=np_rng, theano_rng=theano_rng, hidden_layers_sizes=hidden_layers,
                            n_ins=input_size, n_outs=output_size,
                            input_dropout_factor=0.0, dropout_factor=0.0,
                            dnn_shared=dnn_shared, shared_layers=shared_layers)
        else:
            dnn = DNNDropout(np_rng=np_rng, theano_rng=theano_rng, hidden_layers_sizes=hidden_layers,
                            n_ins=input_size, n_outs=output_size,
                            input_dropout_factor=0.1, dropout_factor=0.5)
        # add dnn and the functions to the list
        dnn_array.append(dnn)

    # # consider the tasks which have nonzero learning rate
    # active_tasks = [n for n in xrange(num_tasks)]

    log('> ... bootstrapping all tasks datasets and building the functions')

    # keep track of the training error in order to create the train/validation
    # curve
    epoch_train_error_array = [[] for n in xrange(num_tasks)]
    epoch_counter = 0

    # keep making bootstraps yo
    while(True):
        inp, outp = ([[1, 1], [0, 1], [1, 0], [0, 0]], [[0, 1, 1, 0], [1, 0, 0, 1]])
        inp = np.array(inp)
        outp = np.array(outp)
        # create new function arrays for the respective bootstrap
        valid_fn_array = []
        test_fn_array = []

        # this array holds the training errors per minibatch
        epoch_train_error_array = [[] for n in xrange(num_tasks)]

        log('> ... building functions for bootstrap found %d' % epoch_counter)
        # build the finetuning functions for these bootstraps
        for idx, task in enumerate(dnn_array):
            train_fn, valid_fn, test_fn = dnn.build_functions(
                (inp[idx], outp[idx]), ([], []), ([], []), mbatch_size, onlyTrain=True)
            train_fn_array.append(train_fn)

        total_train_err = 0.0
        total_cost = 0.0
        # now we're going to train
        for taskidx in xrange(num_tasks):
            for batchidx in xrange(mbatch_per_bootstrap):
                one_err, one_cost = train_fn_array[taskidx](index=batchidx, learning_rate=train_learning_rate)
                one_err = float(one_err)
                total_cost += one_cost
                epoch_train_error_array[taskidx].append(one_err)
                batch_test_err = test_fn_array[taskidx](index=batchidx)
                if not math.isnan(batch_test_err): 
                    test_err += batch_test_err
            mean_train_err = np.mean(epoch_train_error_array[taskidx])
            log('> task %d, bootstrap round %d, training error %f ' % (
                taskidx, epoch_counter, 100 * mean_train_err) + '(%)')
            total_train_err += mean_train_err

        mean_train_error_array.append(total_train_err / num_tasks)

        log('> bootstrap round %d, average cost %f ' % (
            epoch_counter, total_cost / num_tasks))

        log('> bootstrap round %d, Mean training error %f ' % (
            epoch_counter, 100 * total_train_err / float(num_tasks)))
        # increment the epoch counter
        epoch_counter += 1
