import numpy as np
import sys
import datetime

from theano.tensor.shared_randomstreams import RandomStreams

from network.model.dropout_dnn import DNNDropout
from util.load_data import get_minibatches

shared_layers_sizes = [512, 512]
task_specific_sizes = [[512, 512]] * 20

input_size = 4096
output_size = 2

num_bootstrap_rds = 500

mbatch_size = 200
bootstrap_size = 10000
mbatch_per_bootstrap = bootstrap_size / mbatch_size

num_tasks = len(task_specific_sizes)
shared_layers_num = len(shared_layers_sizes)


def log(string):

    sys.stderr.write('[' + str(datetime.now()) + '] ' + str(string) + '\n')

if __name__ == '__main__':

    # we keep track of the training fns, valid fns, and networks
    train_fn_array = []
    valid_fn_array = []
    dnn_array = []

    np_rng = np.random.RandomState(89677)
    theano_rng = RandomStreams(np_rng.randint(2 ** 30))

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

    # build the validation minibatch
    val_tasks, val_x, val_y = np.ndarray(0), None, np.ndarray(0)
    for task, inp, outp in get_minibatches(batch_size=1000, num_epochs=1, add_bias=True):
        val_tasks = np.hstack((val_tasks, task.ravel()))
        val_y = np.hstack((val_y, outp.ravel()))
        if val_x is None:
            val_x = inp
        else:
            val_x = np.vstack((val_x, inp))

    # # consider the tasks which have nonzero learning rate
    # active_tasks = [n for n in xrange(num_tasks)]
    train_error_array = [[] for n in xrange(num_tasks)]

    # BOOTSTRAP 20 DATASETS OF SIZE batch_size
    for cnt, (task, inp, outp) in enumerate(get_minibatches(
            batch_size=bootstrap_size, num_epochs=num_bootstrap_rds, add_bias=True)):

        log('> ... bootstrapping all tasks datasets and building the functions')
        train_fn_array = []
        valid_fn_array = []

        # build the finetuning functions for these bootstraps
        for idx, task in enumerate(dnn_array):
            train_fn, valid_fn = dnn.build_functions(
                (inp[idx], outp[idx]), (val_x, val_y), mbatch_size)
            train_fn_array.append(train_fn)
            valid_fn_array.append(valid_fn)

        # now we're going to train for 100 epochs per bootstrap
        for taskidx in xrange(num_tasks):
            for batchidx in xrange(mbatch_per_bootstrap):
                train_error_array[taskidx].append(
                    train_fn_array[taskidx](index=batchidx))
            log('> task %d, epoch %d, training error %f ' % (
                taskidx, "6969.00", 100 * np.mean(train_error_array[n])) + '(%)')

        # # we validate after we finish one bootstrap
        # valid_error = validate_by_minibatch(valid_fn_array[n], cfg)
        #     log('> task %d, epoch %d, lrate %f, validation error %f ' % (
        # n, cfg.lrate.epoch, cfg.lrate.get_rate(), 100 *
        # numpy.mean(valid_error)) + '(%)')
