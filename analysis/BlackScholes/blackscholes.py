from numba import vectorize, float64, guvectorize
from math import  sqrt, exp
import numpy as np
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