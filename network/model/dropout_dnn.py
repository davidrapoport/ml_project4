import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from network.layers.base_hiddenlayer import DropoutHiddenLayer, HiddenLayer, _dropout_from_layer
from network.layers.logreg_layer import LogisticRegression
import collections


class DNNDropout(object):

    def __init__(self, np_rng, hidden_layers_sizes, n_ins, n_outs, theano_rng=None,
                 dnn_shared=None, shared_layers=[], input_dropout_factor=0.5, dropout_factor=0.5,
                 ):

        self.layers = []
        self.dropout_layers = []
        self.params = []
        self.delta_params = []

        self.n_ins = n_ins
        self.n_outs = n_outs
        self.hidden_layers_sizes = hidden_layers_sizes
        self.hidden_layers_number = len(self.hidden_layers_sizes)

        self.input_dropout_factor = input_dropout_factor
        self.dropout_factor = dropout_factor

        # sometimes you need a theano rng and not a numpy one
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        for i in xrange(self.hidden_layers_number):
            # construct the hidden layer
            if i == 0:
                input_size = self.n_ins
                layer_input = self.x
                if self.input_dropout_factor > 0.0:
                    dropout_layer_input = _dropout_from_layer(
                        theano_rng, self.x, self.input_dropout_factor)
                else:
                    dropout_layer_input = self.x
            else:
                input_size = self.hidden_layers_sizes[i - 1]
                layer_input = (1 - self.dropout_factor) * \
                    self.layers[-1].output
                dropout_layer_input = self.dropout_layers[-1].dropout_output

            W = None
            b = None
            if (i in shared_layers):
                W = dnn_shared.layers[i].W
                b = dnn_shared.layers[i].b

            dropout_layer = DropoutHiddenLayer(rng=np_rng,
                                               input=dropout_layer_input,
                                               n_in=input_size,
                                               n_out=self.hidden_layers_sizes[
                                                   i],
                                               W=W, b=b,
                                               dropout_factor=self.dropout_factor)
            hidden_layer = HiddenLayer(rng=np_rng,
                                       input=layer_input,
                                       n_in=input_size,
                                       n_out=self.hidden_layers_sizes[i],
                                       W=dropout_layer.W, b=dropout_layer.b)

            # add the layer to our list of layers
            self.layers.append(hidden_layer)
            self.dropout_layers.append(dropout_layer)
            self.params.extend(dropout_layer.params)
            self.delta_params.extend(dropout_layer.delta_params)

        # We now need to add a logistic layer on top of the MLP
        self.dropout_logLayer = LogisticRegression(
            input=self.dropout_layers[-1].dropout_output,
            n_in=self.hidden_layers_sizes[-1], n_out=self.n_outs)

        self.logLayer = LogisticRegression(
            input=(1 - self.dropout_factor) * self.layers[-1].output,
            n_in=self.hidden_layers_sizes[-1], n_out=self.n_outs,
            W=self.dropout_logLayer.W, b=self.dropout_logLayer.b)

        self.dropout_layers.append(self.dropout_logLayer)
        self.layers.append(self.logLayer)
        self.params.extend(self.dropout_logLayer.params)
        self.delta_params.extend(self.dropout_logLayer.delta_params)

        # compute the cost
        self.finetune_cost = self.dropout_logLayer.negative_log_likelihood(
            self.y)
        self.errors = self.logLayer.errors(self.y)

    def build_functions(self, train_shared_xy, valid_shared_xy, batch_size):

        (train_set_x, train_set_y) = train_shared_xy
        (valid_set_x, valid_set_y) = valid_shared_xy

        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.fscalar('learning_rate')
        momentum = T.fscalar('momentum')

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = collections.OrderedDict()
        for dparam, gparam in zip(self.delta_params, gparams):
            updates[dparam] = momentum * dparam - gparam * learning_rate
        for dparam, param in zip(self.delta_params, self.params):
            updates[param] = param + updates[dparam]

        train_fn = theano.function(inputs=[index, theano.Param(learning_rate, default=0.0001),
                                           theano.Param(momentum, default=0.5)],
                                   outputs=self.errors,
                                   updates=updates,
                                   givens={
            self.x: train_set_x[index * batch_size:
                                (index + 1) * batch_size],
            self.y: train_set_y[index * batch_size:
                                (index + 1) * batch_size]})

        valid_fn = theano.function(inputs=[index],
                                   outputs=self.errors,
                                   givens={
            self.x: valid_set_x[index * batch_size:
                                (index + 1) * batch_size],
            self.y: valid_set_y[index * batch_size:
                                (index + 1) * batch_size]})

        return train_fn, valid_fn
