import numpy as np
from matplotlib import pyplot
from numba.experimental import jitclass
from numba import njit, float64, int64, types
import warnings
import time

from utils.helpers import generate_consecutive_period_sample

warnings.simplefilter(action='ignore', category=FutureWarning)


class Distribution:
    def __init__(self):
        self.low_dist = None
        self.high_dist = None
        self.close_dist = None
        self.total = None


class DeltaProbabilityStats:
    def __init__(self):
        self.close = None
        self.high = None
        self.low = None


class Ticker:
    def __init__(self, data, period=5):
        self._range_data = data
        self._open_data = None
        self._close_data_delta = None
        self._min_data_delta = None
        self._max_data_delta = None
        self._total = None
        self._envelope = None
        self._period = period
        self._distribution_full = Distribution()
        self.calculate_basic_stats()
        low_dist, high_dist, close_dist = Ticker.__calculate_distribution(self._total, self._close_data_delta, self._min_data_delta, self._max_data_delta)
        self._distribution_full.close_dist = close_dist
        self._distribution_full.low_dist = low_dist
        self._distribution_full.high_dist = high_dist
        self._distribution_full.total = self._close_data_delta.shape[0]

    def calculate_basic_stats(self):
        # working_data = self._range_data
        # open_data = working_data[:-period]

        working_period = self._period + 1
        samples = round(self._range_data.shape[0]/working_period)
        random_sample_idx_arr = np.round((np.random.rand(samples)) * (self._range_data.shape[0] - self._period - 10)).astype('int32')
        working_data = generate_consecutive_period_sample(random_sample_idx_arr, self._range_data, working_period)
        self._close_data_delta = working_data[:, 0]
        self._min_data_delta = working_data[:, 1]
        self._max_data_delta = working_data[:, 2]
        # print(working_data[10])

        # self._close_data_delta = (close_data - open_data)/open_data
        # # self._close_data_delta = working_data[period:]
        #
        # min_data = np.lib.stride_tricks.sliding_window_view(working_data[1:], window_shape=period)
        # self._min_data_delta = (np.min(min_data, axis=1) - open_data) / open_data
        #
        # max_data = np.lib.stride_tricks.sliding_window_view(working_data[1:], window_shape=period)
        # self._max_data_delta = (np.max(max_data, axis=1) - open_data) / open_data

        # self._close_data_delta = (self._close_data_delta - open_data) / open_data
        # self._open_data = open_data
        self._total = self._close_data_delta.shape[0]

    @staticmethod
    @njit([types.containers.UniTuple(types.containers.Tuple([int64[:], float64[:]]), 3)(float64, float64[:], float64[:], float64[:])], cache=True)
    def __calculate_distribution(total, close_data_delta, min_data_delta, max_data_delta):
        bin_count = 40
        low_dist = np.histogram(min_data_delta[-1*total:], bin_count)
        high_dist = np.histogram(max_data_delta[-1*total:], bin_count)
        close_dist = np.histogram(close_data_delta[-1*total:], bin_count)

        return low_dist, high_dist, close_dist

    def __calculate_distribution_nrs(self):
        bin_count = 40
        distribution = Distribution()
        distribution.total = self._total
        distribution.low_dist = np.histogram(self._min_data_delta[-1*distribution.total:], bin_count)
        distribution.high_dist = np.histogram(self._max_data_delta[-1*distribution.total:], bin_count)
        distribution.close_dist = np.histogram(self._close_data_delta[-1*distribution.total:], bin_count)

        return distribution

    def calculate_envelopes(self):
        envelope_min = np.histogram(self._min_data_delta, 10, range=(-0.4, 0.59))
        envelope_max = np.histogram(self._max_data_delta, 10, range=(-0.4, 0.59))
        envelope_close = np.histogram(self._close_data_delta, 10, range=(-0.4, 0.59))
        min = np.array([envelope_min[0][:3], envelope_min[0][4]])/self._total
        max = np.array([envelope_max[0][3], envelope_max[0][4]])/self._total
        close = np.array([envelope_close[0][3], envelope_close[0][4]])/self._total
        print("")



    @staticmethod
    @njit([types.containers.Tuple([float64, float64, float64])(float64, types.containers.Tuple([int64[:], float64[:]]), types.containers.Tuple([int64[:], float64[:]]), types.containers.Tuple([int64[:], float64[:]]), int64)], cache=True)
    def _calc_probability_dist_internal(delta, close_dist, low_dist, high_dist, total):
        if delta > 0:
            close = np.nonzero(close_dist[1] >= delta)
            low = np.nonzero(low_dist[1] >= delta)
            high = np.nonzero(high_dist[1] >= delta)
        else:
            close = np.nonzero(close_dist[1] <= delta)
            low = np.nonzero(low_dist[1] <= delta)
            high = np.nonzero(high_dist[1] <= delta)

        c = (np.sum(close_dist[0][close])/total)
        l = np.sum(low_dist[0][low])/total
        h = np.sum(high_dist[0][high])/total

        return c, l, h

    @staticmethod
    def _calc_probability_dist_nrs(delta, distribution):
        if delta > 0:
            close = [x - 1 for x in np.nonzero(distribution.close_dist[1] >= delta)]
            low = [x - 1 for x in np.nonzero(distribution.low_dist[1] >= delta)]
            high = [x - 1 for x in np.nonzero(distribution.high_dist[1] >= delta)]
        else:
            close = np.nonzero(distribution.close_dist[1] <= delta)
            low = np.nonzero(distribution.low_dist[1] <= delta)
            high = np.nonzero(distribution.high_dist[1] <= delta)

        dps = DeltaProbabilityStats()
        dps.high = np.sum(distribution.high_dist[0][high]) / distribution.total
        dps.low = np.sum(distribution.low_dist[0][low]) / distribution.total
        dps.close = np.sum(distribution.close_dist[0][close]) / distribution.total
        return dps

    def calc_probability(self, delta):

        c, l, h = Ticker._calc_probability_dist_internal(delta, self._distribution_full.close_dist, self._distribution_full.low_dist, self._distribution_full.high_dist, self._distribution_full.total)
        dps = DeltaProbabilityStats()
        dps.close = c
        dps.low = l
        dps.high = h

        return dps
