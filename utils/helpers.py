from numba import njit, vectorize, float64, int32, guvectorize
import numpy as np
from math import fabs, erf, erfc, log, pow, sqrt, pi, exp

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

@njit
def tryr():
    np.histogram(np.ones(100), 10)


def generate_consecutive_period_sample(sample_arr, data_source, period_len):
    len_pdr = np.zeros(3)
    return generate_consecutive_period_sample_internal(len_pdr, sample_arr, data_source, period_len)


@guvectorize([(float64[:], int32[:], float64[:], int32, float64[:, :])], '(p),(m),(n),()->(m,p)', nopython=True, cache=True)
def generate_consecutive_period_sample_internal(three_len, sample_arr, data_source, period_len, result):
    # result = np.zeros((sample_arr.shape[0], period_len))
    for i in range(sample_arr.shape[0]):
        result[i][0] = data_source[sample_arr[i] + period_len - 1]
        result[i][1] = 9999999999999.9
        result[i][2] = -9999999999999.9
        for j in range(1, period_len):
            result[i][1] = min(result[i][1], data_source[sample_arr[i] + j])
            result[i][2] = max(result[i][2], data_source[sample_arr[i] + j])

        open_val = data_source[sample_arr[i]]
        result[i][0] = (result[i][0] - open_val)/open_val
        result[i][1] = (result[i][1] - open_val)/open_val
        result[i][2] = (result[i][2] - open_val)/open_val
    # return result


# @guvectorize([(float64[:], int32[:], float64[:], int32, float64[:, :])], '(p),(m),(n),()->(m,p)', nopython=True, cache=True)
# def generate_consecutive_period_sample_internal(len_phdr, sample_arr, data_source, period_len, result):
#     # result = np.zeros((sample_arr.shape[0], period_len))
#     for i in range(sample_arr.shape[0]):
#         for j in range(period_len):
#             result[i][j] = data_source[sample_arr[i] + j]
#     # return result

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
