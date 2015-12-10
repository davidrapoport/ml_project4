import numpy as np
import sys
import datetime

from theano.tensor.shared_randomstreams import RandomStreams

from model.dropout_dnn import DNNDropout

shared_layers_sizes = [512, 512]
task_specific_sizes = [[512, 512]] * 20

input_size = 4096
output_size = 2

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

        # get the training and validation functions for this task
        # train_fn, valid_fn = dnn.build_functions(stuff), batch_size =
        # batch_size))

        # add dnn and the functions to the list
        dnn_array.append(dnn)
        log('> ... building the model for task %d' % (n))
        # train_fn_array.append(train_fn)
        # valid_fn_array.append(valid_fn)
