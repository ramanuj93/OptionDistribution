import numpy as np
import timeit
import matplotlib.pyplot as plt
from numba import vectorize, float64, guvectorize
from math import  sqrt, exp
from utils.helpers import normal_dist, black_scholes_d1, black_scholes_d2, pdf


class BlackScholes:
    @staticmethod
    @guvectorize([(float64[:], float64[:], float64, float64[:], float64[:], float64[:], float64[:, :, :, :, :])], '(m),(s),(),(t),(r),(v)->(s,t,r,v,m)', target='parallel', fastmath=True)
    def _get_greeks_call(g_types, S, K, tte, rates, stdiv, result):
        for r in range(rates.shape[0]):
            for t in range(tte.shape[0]):
                np_sqrt_t = sqrt(tte[t])
                exp_ert = exp(-1*rates[r]*tte[t])
                for s in range(S.shape[0]):
                    for v in range(stdiv.shape[0]):
                        # intermediate steps
                        d1 = black_scholes_d1(S[s], K, tte[t], rates[r], stdiv[v])
                        d2 = black_scholes_d2(d1, tte[t], stdiv[v])
                        pdf_d1 = pdf(d1)
                        cdf_d1 = normal_dist(d1)
                        cdf_d2 = normal_dist(d2)

                        # final calculations
                        price = (S[s]*cdf_d1)-(K*exp_ert*cdf_d2)
                        delta = cdf_d1  # calculation of delta
                        gamma = (1/(S[s]*stdiv[v]*np_sqrt_t))*pdf_d1  # calculation of gamma
                        theta = (1/365.25)*(((-1)*(S[s]*stdiv[v]*pdf_d1/(2*np_sqrt_t))) - (r*K*exp_ert*cdf_d2))  # calculation of theta

                        # assignment
                        result[s][t][r][v][0] = price * 100.0
                        result[s][t][r][v][1] = delta * 100.0
                        result[s][t][r][v][2] = gamma * 100.0
                        result[s][t][r][v][3] = theta * 100.0


    @staticmethod
    @guvectorize([(float64[:], float64[:], float64, float64[:], float64[:], float64[:], float64[:, :, :, :, :])], '(m),(s),(),(t),(r),(v)->(s,t,r,v,m)', target='parallel', fastmath=True)
    def _get_greeks_put(g_types, S, K, tte, rates, stdiv, result):
        for r in range(rates.shape[0]):
            for t in range(tte.shape[0]):
                np_sqrt_t = sqrt(tte[t])
                exp_ert = exp(-1*rates[r]*tte[t])
                for s in range(S.shape[0]):
                    for v in range(stdiv.shape[0]):
                        # intermediate steps
                        d1 = black_scholes_d1(S[s], K, tte[t], rates[r], stdiv[v])
                        d2 = black_scholes_d2(d1, tte[t], stdiv[v])
                        pdf_d1 = pdf(d1)
                        cdf_d1 = normal_dist(d1)

                        # final calculations
                        price = (K*exp_ert*normal_dist(-1*d2))-(S[s] * normal_dist(-1*d1))
                        delta = cdf_d1 - 1  # calculation of delta
                        gamma = (1/(S[s]*stdiv[v]*np_sqrt_t))*pdf_d1  # calculation of gamma
                        theta = (1/365.25) * (((-1)*(S[s] * stdiv[v] * pdf_d1/(2 * np_sqrt_t))) + (rates[r] * K * exp_ert * normal_dist(-1*d2)))  # calculation of theta

                        # assignment
                        result[s][t][r][v][0] = price * 100.0
                        result[s][t][r][v][1] = delta * 100.0
                        result[s][t][r][v][2] = gamma * 100.0
                        result[s][t][r][v][3] = theta * 100.0

    @staticmethod
    def get_greeks_call(S, K, t, r, stdiv):
        """
        Returns price and greeks for call option(s)
                Parameters:
                        S: Price(s) of Underlying
                        K: Strike Price
                        t: Time(s) to Expiry in Days
                        r: Risk Free Rate(s)
                        stdiv: Standard Deviation(s)

                Returns:
                        Price And Greeks: Price, Delta, Gamma, Theta
        """
        g_types = np.ones(4)
        underlying_prices = None
        tte = None
        rates1 = None
        st_divs = None

        if type(S) is not np.ndarray:
            underlying_prices = np.ones(1)*S
        else:
            underlying_prices = S

        if type(t) is not np.ndarray:
            tte = np.ones(1)*t/365.25
        else:
            tte = t/365.25

        if type(r) is not np.ndarray:
            rates1 = np.ones(1)*r
        else:
            rates1 = r

        if type(stdiv) is not np.ndarray:
            st_divs = np.ones(1)*stdiv
        else:
            st_divs = stdiv

        return BlackScholes._get_greeks_call(g_types, underlying_prices, K, tte, rates1, st_divs)

    @staticmethod
    def get_greeks_put(S, K, t, r, stdiv):
        """
        Returns price and greeks for put option(s)
                Parameters:
                        S: Price(s) of Underlying
                        K: Strike Price
                        t: Time(s) to Expiry in Days
                        r: Risk Free Rate(s)
                        stdiv: Standard Deviation(s)

                Returns:
                        Price And Greeks: Price, Delta, Gamma, Theta
        """
        g_types = np.ones(4)
        underlying_prices = None
        tte = None
        rates1 = None
        st_divs = None

        if type(S) is not np.ndarray:
            underlying_prices = np.ones(1) * S
        else:
            underlying_prices = S

        if type(t) is not np.ndarray:
            tte = np.ones(1) * t / 365.25
        else:
            tte = t / 365.25

        if type(r) is not np.ndarray:
            rates1 = np.ones(1) * r
        else:
            rates1 = r

        if type(stdiv) is not np.ndarray:
            st_divs = np.ones(1) * stdiv
        else:
            st_divs = stdiv

        return BlackScholes._get_greeks_put(g_types, underlying_prices, K, tte, rates1, st_divs)


class Optionleg:
    def __init__(self, K, type, tte, cb):
        self.cost_basis = cb
        self.is_closed = False
        self.K = K
        self.type = type
        self.tte = tte
        self.price = None
        self.delta = None
        self.gamma = None
        self.theta = None

    def calculate_domain(self, curr_S, rf_rate, stdiv):
        if self.type == 'c':
            K = self.K
            S = np.concatenate((np.ones(1)*curr_S, np.linspace(curr_S*0.7, curr_S*1.3, 61)))
            t = np.linspace(self.tte, 1, self.tte)
            r = rf_rate
            v = np.concatenate((np.ones(1)*stdiv, np.linspace(max(0, stdiv - 0.1), stdiv + 0.1, 6)))

            results = BlackScholes.get_greeks_call(S, K, t, r, v)
            self.price = results[0, 0, 0, 0, 0]
            self.delta = results[0, 0, 0, 0, 1]
            self.gamma = results[0, 0, 0, 0, 2]
            self.theta = results[0, 0, 0, 0, 3]
            print("")



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


S = 108.38
K = 100
stdiv1 = 0.38
stdiv2 = 0.375
times = 89
rates2 = 0.03015
start = timeit.default_timer()
leg = Optionleg(130, 'c', 89, 1.0)
leg.calculate_domain(108.38, 0.03015, 0.35)

option_data1 = BlackScholes.get_greeks_put(S, K, times, rates2, stdiv1)
# option_data2 = BlackScholes.get_greeks_call(S, K, times, rates, stdiv2)
print(f'time -> {(timeit.default_timer()-start)*1000}')
# print(option_data1[0,0,0,0])

# plt.plot(times, option_data1[:,:,:,:, -1], color='r')
# plt.plot(times, option_data2[:, -1], color='g')
# plt.show()



# @vectorize([float64(float64, float64, float64, float64, float64)])
# def get_delta(s, k, t, r, sigma):
#     return theta('c', s, k, t, r, sigma)*-100
# print(normal_dist(0.75))
# print(norm._cdf(np.array([0.75]), 0, 1))
# dt = get_delta(S, K, times, rates, stdiv)
# plt.plot(times, dt)
# plt.show()
# print(dt)