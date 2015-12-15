import theano
import theano.tensor as T
import numpy as np
import theano.printing
from theano.tensor.shared_randomstreams import RandomStreams


class HiddenLayer(object):

    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """

        self.input = input
        # initialize the weights
        if W is None:
            W_values = np.random.randn(n_in, n_out).astype(theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        # initialize the bias vector
        # the size of this vector is the number of hidden units in the layer
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        self.delta_W = theano.shared(value = np.zeros((n_in,n_out), dtype=theano.config.floatX), name='delta_W')
        self.delta_b = theano.shared(value = np.zeros_like(self.b.get_value(borrow=True), dtype=theano.config.floatX), name='delta_b')

        # theano symbolic expression to compute w^Tx + b
        lin_output = T.dot(input, self.W) + self.b

        # nonlinearity activation function
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        # parameters of the model
        self.params = [self.W, self.b]
        self.delta_params = [self.delta_W, self.delta_b]


def _dropout_from_layer(theano_rng, hid_out, p):
    """ p is the factor for dropping a unit """
    # p=1-p because 1's indicate keep and p is prob of dropping
    return theano_rng.binomial(n=1, p=1 - p, size=hid_out.shape,
                               dtype=theano.config.floatX) * hid_out


class DropoutHiddenLayer(HiddenLayer):

    def __init__(self, rng, input, n_in, n_out, dropout_factor=0.5, activation=T.tanh, W=None, b=None):

        super(DropoutHiddenLayer, self).__init__(
            rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
            activation=activation)

        self.theano_rng = RandomStreams(rng.randint(2 ** 30))

        self.dropout_output = _dropout_from_layer(theano_rng=self.theano_rng,
                                                  hid_out=self.output, p=dropout_factor)
