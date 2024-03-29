import numpy as np
# import timeit
from numba import guvectorize, float64, int32
import matplotlib.pyplot as plt
from analysis.BlackScholes.blackscholes import BlackScholes
from time import time, time_ns


@guvectorize([(float64[:, :], float64[:], float64, float64[:])], '(p,t),(p),() -> (t)', cache=True, nopython=True)
def get_price_times(price_time_price_vector, prices_vector, cost_basis, result):
    for time_i in range(price_time_price_vector.shape[1]):
        for price_i in range(price_time_price_vector.shape[0]):
            if price_time_price_vector[price_i][time_i] > cost_basis:
                result[time_i] = prices_vector[price_i]
                break


class Optionleg:
    def __init__(self, K, type, tte, cb):
        self.cost_basis = cb * 100.0
        self.is_closed = False
        self.K = K
        self.type = type
        self.tte = tte
        self.S = None
        self.price = None
        self.delta = None
        self.gamma = None
        self.theta = None
        # s, t, r, vol, greeks
        self.state_space = None
        self.underlying_price_domain = None
        self.timeline_expiry = None
        self.vol_domain = None

    def calculate_domain(self, curr_S, rf_rate, stdiv):
        K = self.K
        S = np.concatenate((np.ones(1) * curr_S, np.linspace(curr_S * 0.7, curr_S * 1.3, 61)))
        t = np.linspace(self.tte, 1, self.tte)
        r = rf_rate
        v = np.concatenate((np.ones(1) * stdiv, np.linspace(max(0, stdiv - 0.1), stdiv + 0.1, 6)))

        self.underlying_price_domain = S
        self.timeline_expiry = t
        self.vol_domain = v

        if self.type == 'c':
            results = BlackScholes.get_greeks_call(S, K, t, r, v)
        else:
            results = BlackScholes.get_greeks_put(S, K, t, r, v)

        self.S = curr_S
        # s, t, r, vol, greeks
        self.price = results[0, 0, 0, 0, 0]
        self.delta = results[0, 0, 0, 0, 1]
        self.gamma = results[0, 0, 0, 0, 2]
        self.theta = results[0, 0, 0, 0, 3]
        self.state_space = results[1:, :, :, 1:]

    def get_price_timeline(self):
        # becomes time x price
        price_timeline = (self.state_space[:, :, 0, 0, 0]).transpose()
        return price_timeline


class IronCondor:
    def __init__(self, p1, p2, c1, c2, cb_p1, cb_p2, cb_c1, cb_c2, tte):
        self.long_put = Optionleg(p1, 'p', tte, cb_p1)
        self.short_put = Optionleg(p2, 'p', tte, cb_p2)
        self.short_call = Optionleg(c1, 'c', tte, cb_c1)
        self.long_call = Optionleg(c2, 'c', tte, cb_c2)

    def calculate_domain(self, curr_S, rf_rate, stdiv):
        self.long_put.calculate_domain(curr_S, rf_rate, stdiv)
        self.short_put.calculate_domain(curr_S, rf_rate, stdiv)
        self.short_call.calculate_domain(curr_S, rf_rate, stdiv)
        self.long_call.calculate_domain(curr_S, rf_rate, stdiv)

    def get_breakevens(self):
        price = self.long_put.price - self.short_put.price + self.long_call.price - self.short_call.price
        delta = self.long_put.delta - self.short_put.delta + self.long_call.delta - self.short_call.delta
        gamma = self.long_put.gamma - self.short_put.gamma + self.long_call.gamma - self.short_call.gamma
        theta = self.long_put.theta - self.short_put.theta + self.long_call.theta - self.short_call.theta

        start = time()
        f = get_price_times(self.long_call.state_space[:, :, 0, 3, 0], self.long_call.underlying_price_domain[1:], self.long_call.cost_basis)

        print(f"Time taken -> {time() - start}")
        y = 9

# np.argwhere(self.long_call.state_space[:,:,0,3,0]>self.long_call.cost_basis)

S = 108.38
K = 100
stdiv1 = 0.38
stdiv2 = 0.375
times = 89
rates2 = 0.03015
leg = Optionleg(130, 'c', 89, 1.0)
start = time()
# leg.calculate_domain(108.38, 0.03015, 0.35)
# leg.get_price_timeline()

option_data1 = BlackScholes.get_greeks_call(S, K, times, rates2, stdiv1)
option_data1 = BlackScholes.get_greeks_put(S, K, times, rates2, stdiv1)
# option_data2 = BlackScholes.get_greeks_call(S, K, times, rates, stdiv2)
# print(f'time -> {time() - start}')
# print(option_data1[0, 0, 0, 0])

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
