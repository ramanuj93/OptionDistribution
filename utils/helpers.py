from numba import njit, vectorize, float64
import numpy as np
from math import fabs, erf, erfc, log, pow, sqrt,pi,exp

NPY_SQRT1_2 = 1.0 / sqrt(2)


@njit(cache=True, fastmath=True)
def normal_dist(a):
    if np.isnan(a):
        return np.nan

    x = a * NPY_SQRT1_2
    z = fabs(x)

    if z < NPY_SQRT1_2:
        y = 0.5 + 0.5 * erf(x)
    else:
        y = 0.5 * erfc(z)

        if x > 0:
            y = 1.0 - y

    return y


@njit(cache=True, fastmath=True)
def pdf(d1):
    if np.isnan(d1):
        return np.nan

    return (1/sqrt(2*pi))*exp(-1*pow(d1, 2)/2)


@vectorize([float64(float64, float64, float64, float64, float64)], target='cpu')
def black_scholes_d1(S, K, t, r, stdiv):
    return (log(S/K) + t*(r + (pow(stdiv, 2)/2)))/(stdiv*sqrt(t))


@vectorize([float64(float64, float64, float64)])
# @njit(cache=True, fastmath=True)
def black_scholes_d2(d1, t, stdiv):
    return d1 - (stdiv*sqrt(t))
