import cPickle as pkl
import time

import numpy
import theano
from theano import config
import theano.tensor as T
from theano.tensor.nnet import categorical_crossentropy

from fuel.datasets import TextFile
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme
from fuel.transformers import Batch, Padding


# These files can be downloaded from
# http://www-etud.iro.umontreal.ca/~brakelp/train.txt.gz
# http://www-etud.iro.umontreal.ca/~brakelp/dictionary.pkl
# don't forget to change the paths and gunzip train.txt.gz
TRAIN_FILE = '/u/brakelp/temp/traindata.txt'
VAL_FILE = '/u/brakelp/temp/valdata.txt'
DICT_FILE = '/u/brakelp/temp/dictionary.pkl'


def sequence_categorical_crossentropy(prediction, targets, mask):
    prediction_flat = prediction.reshape(((prediction.shape[0] *
                                           prediction.shape[1]),
                                          prediction.shape[2]), ndim=2)
    targets_flat = targets.flatten()
    mask_flat = mask.flatten()
    ce = categorical_crossentropy(prediction_flat, targets_flat)
    return T.sum(ce * mask_flat)


def gauss_weight(ndim_in, ndim_out=None, sd=.005):
    if ndim_out is None:
        ndim_out = ndim_in
    W = numpy.random.randn(ndim_in, ndim_out) * sd
    return numpy.asarray(W, dtype=config.floatX)


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                               name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        energy = T.dot(input, self.W) + self.b
        energy_exp = T.exp(energy - T.max(energy, 2)[:, :, None])
        pmf = energy_exp / energy_exp.sum(2)[:, :, None]
        self.p_y_given_x = pmf

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]


def index_dot(indices, w):
    return w[indices.flatten()]


class LstmLayer:

    def __init__(self, rng, input, mask, n_in, n_h):

        # Init params
        self.W_i = theano.shared(gauss_weight(n_in, n_h), 'W_i', borrow=True)
        self.W_f = theano.shared(gauss_weight(n_in, n_h), 'W_f', borrow=True)
        self.W_c = theano.shared(gauss_weight(n_in, n_h), 'W_c', borrow=True)
        self.W_o = theano.shared(gauss_weight(n_in, n_h), 'W_o', borrow=True)

        self.U_i = theano.shared(gauss_weight(n_h), 'U_i', borrow=True)
        self.U_f = theano.shared(gauss_weight(n_h), 'U_f', borrow=True)
        self.U_c = theano.shared(gauss_weight(n_h), 'U_c', borrow=True)
        self.U_o = theano.shared(gauss_weight(n_h), 'U_o', borrow=True)

        self.b_i = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                 'b_i', borrow=True)
        self.b_f = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                 'b_f', borrow=True)
        self.b_c = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                 'b_c', borrow=True)
        self.b_o = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                 'b_o', borrow=True)

        self.params = [self.W_i, self.W_f, self.W_c, self.W_o,
                       self.U_i, self.U_f, self.U_c, self.U_o,
                       self.b_i, self.b_f, self.b_c, self.b_o]

        outputs_info = [T.zeros((input.shape[1], n_h)),
                        T.zeros((input.shape[1], n_h))]

        rval, updates = theano.scan(self._step,
                                    sequences=[mask, input],
                                    outputs_info=outputs_info)

        # self.output is in the format (batchsize, n_h)
        self.output = rval[0]

    def _step(self, m_, x_, h_, c_):

        i_preact = (index_dot(x_, self.W_i) +
                    T.dot(h_, self.U_i) + self.b_i)
        i = T.nnet.sigmoid(i_preact)

        f_preact = (index_dot(x_, self.W_f) +
                    T.dot(h_, self.U_f) + self.b_f)
        f = T.nnet.sigmoid(f_preact)

        o_preact = (index_dot(x_, self.W_o) +
                    T.dot(h_, self.U_o) + self.b_o)
        o = T.nnet.sigmoid(o_preact)

        c_preact = (index_dot(x_, self.W_c) +
                    T.dot(h_, self.U_c) + self.b_c)
        c = T.tanh(c_preact)

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * T.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c


def train_model(batch_size=100, n_h=50, n_epochs=40):

    # Load the datasets with Fuel
    dictionary = pkl.load(open(DICT_FILE, 'r'))
    dictionary['~'] = len(dictionary)
    reverse_mapping = dict((j, i) for i, j in dictionary.items())

    print("Loading the data")
    train = TextFile(files=[TRAIN_FILE],
                     dictionary=dictionary,
                     unk_token='~',
                     level='character',
                     preprocess=str.lower,
                     bos_token=None,
                     eos_token=None)

    train_stream = DataStream.default_stream(train)

    # organize data in batches and pad shorter sequences with zeros
    train_stream = Batch(train_stream,
                         iteration_scheme=ConstantScheme(batch_size))
    train_stream = Padding(train_stream)

    # idem dito for the validation text
    val = TextFile(files=[VAL_FILE],
                     dictionary=dictionary,
                     unk_token='~',
                     level='character',
                     preprocess=str.lower,
                     bos_token=None,
                     eos_token=None)

    val_stream = DataStream.default_stream(val)

    # organize data in batches and pad shorter sequences with zeros
    val_stream = Batch(val_stream,
                         iteration_scheme=ConstantScheme(batch_size))
    val_stream = Padding(val_stream)

    print('Building model')

    # Set the random number generator' seeds for consistency
    rng = numpy.random.RandomState(12345)

    x = T.lmatrix('x')
    mask = T.matrix('mask')

    # Construct the LSTM layer
    recurrent_layer = LstmLayer(rng=rng, input=x, mask=mask, n_in=111, n_h=n_h)

    logreg_layer = LogisticRegression(input=recurrent_layer.output[:-1],
                                      n_in=n_h, n_out=111)

    cost = sequence_categorical_crossentropy(logreg_layer.p_y_given_x,
                                             x[1:],
                                             mask[1:]) / batch_size

    # create a list of all model parameters to be fit by gradient descent
    params = logreg_layer.params + recurrent_layer.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # update_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    learning_rate = 0.1
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    update_model = theano.function([x, mask], cost, updates=updates)

    evaluate_model = theano.function([x, mask], cost)

    # Define and compile a function for generating a sequence step by step.
    x_t = T.iscalar()
    h_p = T.vector()
    c_p = T.vector()
    h_t, c_t = recurrent_layer._step(T.ones(1), x_t, h_p, c_p)
    energy = T.dot(h_t, logreg_layer.W) + logreg_layer.b

    energy_exp = T.exp(energy - T.max(energy, 1)[:, None])

    output = energy_exp / energy_exp.sum(1)[:, None]
    single_step = theano.function([x_t, h_p, c_p], [output, h_t, c_t])

    start_time = time.clock()

    iteration = 0

    for epoch in range(n_epochs):
        print 'epoch:', epoch

        for x_, mask_ in train_stream.get_epoch_iterator():
            iteration += 1

            cross_entropy = update_model(x_.T, mask_.T)


            # Generate some text after each 20 minibatches
            if iteration % 40 == 0:
                try:
                    prediction = numpy.ones(111, dtype=config.floatX) / 111.0
                    h_p = numpy.zeros((n_h,), dtype=config.floatX)
                    c_p = numpy.zeros((n_h,), dtype=config.floatX)
                    initial = 'the meaning of life is '
                    sentence = initial
                    for char in initial:
                        x_t = dictionary[char]
                        prediction, h_p, c_p = single_step(x_t, h_p.flatten(),
                                                           c_p.flatten())
                    sample = numpy.random.multinomial(1, prediction.flatten())
                    for i in range(450):
                        x_t = numpy.argmax(sample)
                        prediction, h_p, c_p = single_step(x_t, h_p.flatten(),
                                                           c_p.flatten())
                        sentence += reverse_mapping[x_t]
                        sample = numpy.random.multinomial(1, prediction.flatten())
                    print 'LSTM: "' + sentence + '"'
                except ValueError:
                    print 'Something went wrong during sentence generation.'

            if iteration % 40 == 0:
                print 'epoch:', epoch, '  minibatch:', iteration
                val_scores = []
                for x_val, mask_val in val_stream.get_epoch_iterator():
                    val_scores.append(evaluate_model(x_val.T, mask_val.T))
                print 'Average validation CE per sentence:', numpy.mean(val_scores)

    end_time = time.clock()
    print('Optimization complete.')
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    train_model()
