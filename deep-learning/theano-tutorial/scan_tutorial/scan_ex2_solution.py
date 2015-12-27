import theano
import theano.tensor as T
import numpy as np

probabilities = T.vector()
nb_samples = T.iscalar()

rng = T.shared_randomstreams.RandomStreams(1234)


def sample_from_pvect(pvect):
    """ Provided utility function: given a symbolic vector of
    probabilities (which MUST sum to 1), sample one element
    and return its index.
    """
    onehot_sample = rng.multinomial(n=1, pvals=pvect)
    sample = onehot_sample.argmax()
    return sample


def set_p_to_zero(pvect, i):
    """ Provided utility function: given a symbolic vector of
    probabilities and an index 'i', set the probability of the
    i-th element to 0 and renormalize the probabilities so they
    sum to 1.
    """
    new_pvect = T.set_subtensor(pvect[i], 0.)
    new_pvect = new_pvect / new_pvect.sum()
    return new_pvect


def step(p):
    sample = sample_from_pvect(p)
    new_p = set_p_to_zero(p, sample)
    return new_p, sample

output, updates = theano.scan(fn=step,
                              outputs_info=[probabilities, None],
                              n_steps=nb_samples)

modified_probabilities, samples = output

f = theano.function(inputs=[probabilities, nb_samples],
                    outputs=[samples],
                    updates=updates)

# Testing the function
test_probs = np.asarray([0.6, 0.3, 0.1], dtype=theano.config.floatX)
for i in range(10):
    print(f(test_probs, 2))
