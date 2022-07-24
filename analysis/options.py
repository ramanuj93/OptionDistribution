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
    @guvectorize([(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:])], '(m),(n),(n),(n),(n),(n)->(n,m)', target='cpu')
    def _get_greeks_call(g_types, S, K, t, r, stdiv, result):
        d1 = black_scholes_d1(S, K, t, r, stdiv)
        d2 = black_scholes_d2(d1, t, stdiv)
        exp_ert = np.exp(-1*r*t)
        np_sqrt_t = np.sqrt(t)
        pdf_d1 = norm._pdf(d1, 0, 1)
        cdf_d1 = norm._cdf(d1, 0, 1)
        cdf_d2 = norm._cdf(d2, 0, 1)
        price = (S*cdf_d1)-(K*exp_ert*cdf_d2)
        delta = cdf_d1  # calculation of delta
        gamma = (1/(S*stdiv*np_sqrt_t))*pdf_d1  # calculation of gamma
        theta = (1/365.25)*(((-1)*(S*stdiv*pdf_d1/(2*np_sqrt_t))) - (r*K*exp_ert*cdf_d2))  # calculation of theta
        for i in range(S.shape[0]):
            result[i][0] = price[i]*100.0
            result[i][1] = delta[i]*100.0
            result[i][2] = gamma[i]*100.0
            result[i][3] = theta[i]*100.0

    @staticmethod
    @guvectorize([(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:])], '(m),(n),(n),(n),(n),(n)->(n,m)', target='cpu')
    def _get_greeks_put(g_types, S, K, t, r, stdiv, result):
        d1 = black_scholes_d1(S, K, t, r, stdiv)
        d2 = black_scholes_d2(d1, t, stdiv)
        exp_ert = np.exp(-1*r*t)
        np_sqrt_t = np.sqrt(t)
        pdf_d1 = norm._pdf(d1, 0, 1)
        cdf_d1 = norm._cdf(d1, 0, 1)
        price = (K*exp_ert*norm._cdf(-1*d2, 0, 1))-(S*norm._cdf(-1*d1, 0, 1))
        delta = cdf_d1 - 1  # calculation of delta
        gamma = (1/(S*stdiv*np_sqrt_t))*pdf_d1  # calculation of gamma
        theta = (1/365.25)*(((-1)*(S*stdiv*pdf_d1/(2*np_sqrt_t))) + (r*K*exp_ert*norm._cdf(-1*d2, 0, 1)))  # calculation of theta
        for i in range(S.shape[0]):
            result[i][0] = price[i]*100.0
            result[i][1] = delta[i]*100.0
            result[i][2] = gamma[i]*100.0
            result[i][3] = theta[i]*100.0

    @staticmethod
    def get_greeks_call(S, K, t, r, stdiv):
        g_types = np.ones(4)
        return BlackScholes._get_greeks_call(g_types, S, K, t, r, stdiv)

    @staticmethod
    def get_greeks_put(S, K, t, r, stdiv):
        g_types = np.ones(4)
        return BlackScholes._get_greeks_put(g_types, S, K, t, r, stdiv)



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


size = 1
S = np.ones(size)*108.38
K = np.ones(size)*85
stdiv = np.ones(size)*0.44
times = np.ones(size)*146/365.25
rates = np.ones(size)*0.03015
flag = 'c'
start = timeit.default_timer()
xx = BlackScholes.get_greeks_put(S, K, times, rates, stdiv)
print(f'time -> {(timeit.default_timer()-start)*1000}')
print(xx[0])



# @vectorize([float64(float64, float64, float64, float64, float64)])
# def get_delta(s, k, t, r, sigma):
#     return theta('c', s, k, t, r, sigma)*-100
# print(normal_dist(0.75))
# print(norm._cdf(np.array([0.75]), 0, 1))
# dt = get_delta(S, K, times, rates, stdiv)
# plt.plot(times, dt)
# plt.show()
# print(dt)