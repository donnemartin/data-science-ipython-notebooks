import numpy as np


def linear_data_sample(N=40, rseed=0, m=3, b=-2):
    rng = np.random.RandomState(rseed)

    x = 10 * rng.rand(N)
    dy = m / 2 * (1 + rng.rand(N))
    y = m * x + b + dy * rng.randn(N)

    return (x, y, dy)


def linear_data_sample_big_errs(N=40, rseed=0, m=3, b=-2):
    rng = np.random.RandomState(rseed)

    x = 10 * rng.rand(N)
    dy = m / 2 * (1 + rng.rand(N))
    dy[20:25] *= 10
    y = m * x + b + dy * rng.randn(N)

    return (x, y, dy)


def sample_light_curve(phased=True):
    from astroML.datasets import fetch_LINEAR_sample
    data = fetch_LINEAR_sample()
    t, y, dy = data[18525697].T

    if phased:
        P_best = 0.580313015651
        t /= P_best

    return (t, y, dy)
    

def sample_light_curve_2(phased=True):
    from astroML.datasets import fetch_LINEAR_sample
    data = fetch_LINEAR_sample()
    t, y, dy = data[10022663].T

    if phased:
        P_best = 0.61596079804
        t /= P_best

    return (t, y, dy)
    
