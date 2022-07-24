import numpy as np
import timeit
import matplotlib.pyplot as plt
from numba import vectorize, float64, guvectorize
from numba_stats import norm

from utils.helpers import normal_dist, black_scholes_d1, black_scholes_d2, pdf


class BlackScholes:

    @staticmethod
    @guvectorize([(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:])], '(m),(n),(n),(n),(n),(n)->(n,m)', target='parallel')
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
    @guvectorize([(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:])], '(m),(n),(n),(n),(n),(n)->(n,m)', target='parallel')
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
        """
        Returns price and greeks for call option(s)
                Parameters:
                        S: Price(s) of Underlying
                        K: Strike Price(s)
                        t: Time(s) to Expiry in Days
                        r: Risk Free Rate(s)
                        stdiv: Standard Deviation(s)

                Returns:
                        Price And Greeks: Price, Delta, Gamma, Theta
        """
        g_types = np.ones(4)
        arr_len = 1

        if type(S) is np.ndarray:
            arr_len = S.shape[0]
        elif type(K) is np.ndarray:
            arr_len = K.shape[0]
        elif type(t) is np.ndarray:
            arr_len = t.shape[0]
        elif type(r) is np.ndarray:
            arr_len = r.shape[0]
        elif type(stdiv) is np.ndarray:
            arr_len = stdiv.shape[0]

        underlying_prices = np.ones(arr_len)*S
        strike_prices = np.ones(arr_len)*K
        tte = np.ones(arr_len)*(t/365.25)
        rates = np.ones(arr_len)*r
        st_divs = np.ones(arr_len)*stdiv

        return BlackScholes._get_greeks_call(g_types, underlying_prices, strike_prices, tte, rates, st_divs)

    @staticmethod
    def get_greeks_put(S, K, t, r, stdiv):
        """
        Returns price and greeks for put option(s)
                Parameters:
                        S: Price(s) of Underlying
                        K: Strike Price(s)
                        t: Time(s) to Expiry in Days
                        r: Risk Free Rate(s)
                        stdiv: Standard Deviation(s)

                Returns:
                        Price And Greeks: Price, Delta, Gamma, Theta
        """
        g_types = np.ones(4)
        arr_len = 1

        if type(S) is np.ndarray:
            arr_len = S.shape[0]
        elif type(K) is np.ndarray:
            arr_len = K.shape[0]
        elif type(t) is np.ndarray:
            arr_len = t.shape[0]
        elif type(r) is np.ndarray:
            arr_len = r.shape[0]
        elif type(stdiv) is np.ndarray:
            arr_len = stdiv.shape[0]

        underlying_prices = np.ones(arr_len) * S
        strike_prices = np.ones(arr_len) * K
        tte = np.ones(arr_len) * (t / 365.25)
        rates = np.ones(arr_len) * r
        st_divs = np.ones(arr_len) * stdiv

        return BlackScholes._get_greeks_put(g_types, underlying_prices, strike_prices, tte, rates, st_divs)


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


S = np.linspace(85, 120, 36)#  *108.38
size = S.shape[0]
K = 130
stdiv1 = 0.325
stdiv2 = 0.375
times = 90
rates = 0.03015
flag = 'c'
start = timeit.default_timer()
option_data1 = BlackScholes.get_greeks_call(S, K, times, rates, stdiv1)
option_data2 = BlackScholes.get_greeks_call(S, K, times, rates, stdiv2)
print(f'time -> {(timeit.default_timer()-start)*1000}')
# print(option_data[0])

plt.plot(S, -1*option_data1[:, 0], color='r')
plt.plot(S, -1*option_data2[:, 0], color='g')
plt.show()





# @vectorize([float64(float64, float64, float64, float64, float64)])
# def get_delta(s, k, t, r, sigma):
#     return theta('c', s, k, t, r, sigma)*-100
# print(normal_dist(0.75))
# print(norm._cdf(np.array([0.75]), 0, 1))
# dt = get_delta(S, K, times, rates, stdiv)
# plt.plot(times, dt)
# plt.show()
# print(dt)