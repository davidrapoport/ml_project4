import numpy as np
import sys
import datetime


from theano.tensor.shared_randomstreams import RandomStreams

from network.model.dropout_dnn import DNNDropout
from load_data import get_bootstraps

shared_layers_sizes = [512, 512]
task_specific_sizes = [[512, 512]] * 20

input_size = 4096
output_size = 2

mbatch_size = 50
num_bootstrap_rds = 500

valid_size = 600
valid_mbatch_per_bootstrap = 600 / mbatch_size

bootstrap_size = 1500
mbatch_per_bootstrap = bootstrap_size / mbatch_size

num_tasks = len(task_specific_sizes)
shared_layers_num = len(shared_layers_sizes)


def log(string):

    sys.stderr.write(
        '[' + str(datetime.datetime.now()) + '] ' + str(string) + '\n')


def validate_by_minibatch(valid_fn, idx):

    yield

if __name__ == '__main__':

    # we keep track of the training fns, valid fns, and networks
    train_fn_array = []
    valid_fn_array = []
    dnn_array = []

    np_rng = np.random.RandomState(89773)
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
        dnn = DNNDropout(np_rng=np_rng, theano_rng=theano_rng, hidden_layers_sizes=hidden_layers,
                         n_ins=input_size, n_outs=output_size,
                         dnn_shared=dnn_shared, shared_layers=shared_layers)

        # add dnn and the functions to the list
        dnn_array.append(dnn)

    # # consider the tasks which have nonzero learning rate
    # active_tasks = [n for n in xrange(num_tasks)]

    valin, valout = get_bootstraps(600)

    log('> ... bootstrapping all tasks datasets and building the functions')

    # keep track of the training error in order to create the train/validation
    # curve
    mean_train_error_array = [[] for n in xrange(num_tasks)]
    epoch_train_error_array = [[] for n in xrange(num_tasks)]
    val_error_array = [[] for n in xrange(num_tasks)]
    epoch_counter = 0

    # keep making bootstraps yo
    while(True):
        inp, outp = get_bootstraps(bootstrap_size)

        # create new function arrays for the respective bootstrap
        train_fn_array = []
        valid_fn_array = []

        # this array holds the training errors per minibatch
        epoch_train_error_array = []

        log('> ... building functions for bootstrap found %d' % epoch_counter)
        # build the finetuning functions for these bootstraps
        for idx, task in enumerate(dnn_array):
            train_fn, valid_fn = dnn.build_functions(
                (inp[idx], outp[idx]), (valin[idx], valout[idx]), mbatch_size)
            train_fn_array.append(train_fn)
            valid_fn_array.append(valid_fn)

        # now we're going to train
        for taskidx in xrange(num_tasks):
            for batchidx in xrange(mbatch_per_bootstrap):
                one_err = float(train_fn_array[taskidx](index=batchidx))
                epoch_train_error_array[taskidx].append(one_err)
            mean_train_err = np.mean(epoch_train_error_array[taskidx])
            log('> task %d, bootstrap round %d, training error %f ' % (
                taskidx, epoch_counter, 100 * mean_train_err) + '(%)')
            mean_train_error_array.append(mean_train_err)

        # we validate after we finish one bootstrap
        valid_error = validate_by_minibatch(valid_fn_array[n])
        log('> task %d, bootstrap round %d, validation error %f ' % (
            taskidx, epoch_counter, 100 *
            valid_error + '(%)'))
        val_error_array.append(valid_error)

        # increment the epoch counter
        epoch_counter += 1

        # SAVE THE MODEL IF WE CAN

