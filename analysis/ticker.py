import numpy as np
from matplotlib import pyplot
import warnings
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
        self.calculate_basic_stats(period=period)
        self._distribution_full = self.__calculate_distribution()
        self._distribution_short = self.__calculate_distribution(600)

    def calculate_basic_stats(self, period):
        working_data = self._range_data
        open_data = working_data[:-period]
        self._close_data_delta = working_data[period:]

        min_data = np.lib.stride_tricks.sliding_window_view(working_data[1:], window_shape=period)
        self._min_data_delta = (np.min(min_data, axis=1) - open_data) / open_data

        max_data = np.lib.stride_tricks.sliding_window_view(working_data[1:], window_shape=period)
        self._max_data_delta = (np.max(max_data, axis=1) - open_data) / open_data

        self._close_data_delta = (self._close_data_delta - open_data) / open_data
        self._open_data = open_data
        self._total = self._close_data_delta.shape[0]

    def __calculate_distribution(self, sample_size=0):
        bin_count = 40
        distribution = Distribution()
        distribution.total = sample_size or self._total
        distribution.low_dist = np.histogram(self._min_data_delta[-1*sample_size:], bin_count)
        distribution.high_dist = np.histogram(self._max_data_delta[-1*sample_size:], bin_count)
        distribution.close_dist = np.histogram(self._close_data_delta[-1*sample_size:], bin_count)

        return distribution

    @staticmethod
    def _calc_probability_dist(delta, distribution):
        if delta > 0:
            close = [x - 1 for x in np.nonzero(distribution.close_dist[1] >= delta)]
            low = [x - 1 for x in np.nonzero(distribution.low_dist[1] >= delta)]
            high = [x - 1 for x in np.nonzero(distribution.high_dist[1] >= delta)]
        else:
            close = np.nonzero(distribution.close_dist[1] <= delta)
            low = np.nonzero(distribution.low_dist[1] <= delta)
            high = np.nonzero(distribution.high_dist[1] <= delta)

        dps = DeltaProbabilityStats()
        dps.high = np.sum(distribution.close_dist[0][high])/distribution.total
        dps.low = np.sum(distribution.close_dist[0][low])/distribution.total
        dps.close = np.sum(distribution.close_dist[0][close])/distribution.total
        return dps

    def calc_probability(self, delta):
        res_full = Ticker._calc_probability_dist(delta, self._distribution_full)
        res_short = Ticker._calc_probability_dist(delta, self._distribution_short)
        return res_full, res_short
