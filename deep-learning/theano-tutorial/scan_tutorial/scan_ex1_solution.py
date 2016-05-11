import theano
import theano.tensor as T
import numpy as np

coefficients = T.vector("coefficients")
x = T.scalar("x")
max_coefficients_supported = 10000


def step(coeff, power, prior_value, free_var):
    return prior_value + (coeff * (free_var ** power))

# Generate the components of the polynomial
full_range = T.arange(max_coefficients_supported)
outputs_info = np.zeros((), dtype=theano.config.floatX)

components, updates = theano.scan(fn=step,
                                  sequences=[coefficients, full_range],
                                  outputs_info=outputs_info,
                                  non_sequences=x)

polynomial = components[-1]
calculate_polynomial = theano.function(inputs=[coefficients, x],
                                       outputs=polynomial,
                                       updates=updates)

test_coeff = np.asarray([1, 0, 2], dtype=theano.config.floatX)
print(calculate_polynomial(test_coeff, 3))
