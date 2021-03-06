from numba import njit
import numpy as np
from math import fabs, erf, erfc, log, pow, sqrt,pi,exp

NPY_SQRT1_2 = 1.0/ np.sqrt(2)


@njit(cache=True, fastmath=True)
def normal_dist1(a):
    if np.isnan(a):
        return np.nan

    x = a * NPY_SQRT1_2
    z = np.fabs(x)

    if z < NPY_SQRT1_2:
        y = 0.5 + 0.5 * erf(x)
    else:
        y = 0.5 * erfc(z)

        if x > 0:
            y = 1.0 - y

    return y

@njit(cache=True, fastmath=True)
def normal_dist(a):
    if np.isnan(a):
        return np.nan



@njit(cache=True, fastmath=True)
def pdf(d1):
    if np.isnan(d1):
        return np.nan

    return (1/sqrt(2*pi))*exp(-1*pow(d1, 2)/2)


@njit(cache=True, fastmath=True)
def black_scholes_d1(S, K, t, r, stdiv):
    return (np.log(S/K) + t*(r + (np.power(stdiv, 2)/2)))/(stdiv*np.sqrt(t))


@njit(cache=True, fastmath=True)
def black_scholes_d2(d1, t, stdiv):
    return d1 - (stdiv*np.sqrt(t))