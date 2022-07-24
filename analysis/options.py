from py_vollib.black_scholes.greeks.analytical import theta
from py_vollib.black_scholes.greeks.analytical import delta
from py_vollib.black_scholes.greeks.analytical import gamma
from py_vollib.black_scholes.greeks.analytical import vega
import py_vollib.black_scholes as option_price
import numpy as np
import timeit
import matplotlib.pyplot as plt
from numba import vectorize, float64, guvectorize
from math import fabs, erf, erfc, exp, log, pow, sqrt
from scipy.special import ndtr
from numba_stats import norm

from utils.helpers import normal_dist, black_scholes_d1, black_scholes_d2, pdf


class BlackScholes:

    @staticmethod
    @guvectorize([(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:])], '(m),(n),(n),(n),(n),(n)->(n,m)')
    def get_greeks_call(g_types, S, K, t, r, stdiv, result):
        d1 = black_scholes_d1(S, K, t, r, stdiv)
        d2 = black_scholes_d2(d1, t, stdiv)
        pdf_d1 = norm._pdf(d1, 0, 1)
        cdf_d2 = norm._cdf(d2, 0, 1)
        delta = norm._cdf(d1, 0, 1)
        gamma = (1/(S*stdiv*np.sqrt(t)))*pdf_d1
        theta = (1/365.25)*(((-1)*(S*stdiv*pdf_d1/(2*np.sqrt(t))))-(r*K*np.exp(-1*r*t)*cdf_d2))
        for i in range(S.shape[0]):
            result[i][0] = delta[i]
            result[i][1] = gamma[i]
            result[i][2] = theta[i]

    @staticmethod
    @vectorize([float64(float64, float64, float64, float64, float64)])
    def call_delta(S, K, t, r, stdiv):
        return normal_dist(black_scholes_d1(S, K, t, r, stdiv))

    @staticmethod
    @vectorize([float64(float64, float64, float64, float64, float64)])
    def put_delta(S, K, t, r, stdiv):
        return normal_dist(black_scholes_d1(S, K, t, r, stdiv)) - 1

    # @staticmethod
    # @vectorize([float64(float64, float64, float64, float64, float64)])
    # def get_gamma(S, d1, t, stdiv):
    #     return (1/(S*stdiv*sqrt(t)))*pdf(d1)


class Optionleg:
    def __init__(self):
        self.cost_basis = None
        self.is_closed = False
        self.K = None
        self.type = None
        self.tte = None


class IronCondor:
    def __init__(self, p1, p2, c1, c2, cb_p1, cb_p2, cb_c1, cb_c2, tte):
        self.long_put = Optionleg()
        self.long_put.cost_basis = cb_p1
        self.long_put.K = p1
        self.short_put = Optionleg()
        self.short_put.cost_basis = cb_p2
        self.short_put.K = p2
        self.long_put.type = self.short_put.type = 'p'
        self.short_call = Optionleg()
        self.short_call.cost_basis = cb_c1
        self.short_call.K = c1
        self.long_call = Optionleg()
        self.short_call.cost_basis = cb_c2
        self.short_call.K = c2
        self.long_call.type = self.short_call.type = 'c'
        self.long_call.tte = self.short_call.tte = self.long_put.tte = self.short_put.tte = tte


S = np.ones(1)*260.28
K = np.ones(1)*300
stdiv = np.ones(1)*0.3
times = np.ones(1)*146/365.25
rates = np.ones(1)*0.03015
flag = 'c'
start = timeit.default_timer()
xx = BlackScholes.get_greeks_call(np.ones(3), S, K, times, rates, stdiv)
print(f'time -> {(timeit.default_timer()-start)*1000}')
print(xx)



# @vectorize([float64(float64, float64, float64, float64, float64)])
# def get_delta(s, k, t, r, sigma):
#     return theta('c', s, k, t, r, sigma)*-100
# print(normal_dist(0.75))
# print(norm._cdf(np.array([0.75]), 0, 1))
# dt = get_delta(S, K, times, rates, stdiv)
# plt.plot(times, dt)
# plt.show()
# print(dt)