# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from analysis.options import IronCondor
from analysis.ticker import Ticker
from ibkrInit import IbkrApp
import numpy as np


from matplotlib import pyplot

data_bins = np.linspace(-0.5, 0.5, 21)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
def main_ibkr():
    app = IbkrApp()
    # ! [connect]
    app.connect("127.0.0.1", 4001, clientId=0)
    # ! [connect]
    print("serverVersion:%s connectionTime:%s" % (app.serverVersion(),
                                                  app.twsConnectionTime()))

    # ! [clientrun]
    app.start()
    wow = app.GetHistoricalData()
    app.run()
    # dat = app.reqId2nReq
    print("okay")


def readStock(ticker):
    return np.genfromtxt(f'data\\{ticker}.csv', delimiter=',')[1:, 1:]


def get_range_diff(np_data, data_range):
    delta = data_range - 1
    return (np_data[delta:] - np_data[:-delta]) / np_data[:-delta]


def get30daydelta(ticker):
    ticker_data = readStock(ticker)
    price_data = ticker_data[:, 4]
    price_delta = get_range_diff(price_data, 30)
    value_bins = np.digitize(price_delta, data_bins, False)
    return price_delta


def get30daydance(ticker):
    ticker_data = readStock(ticker)
    price_data = ticker_data[:, 4]
    windowed_data = np.lib.stride_tricks.sliding_window_view(price_data, window_shape=30)
    relative_var = (windowed_data - windowed_data[:, 0].reshape(-1, 1)) / windowed_data[:, 0].reshape(-1, 1)
    datax = np.abs(relative_var)
    dataz = (datax < 0.05)
    dataz = np.count_nonzero(dataz, axis=1)
    print("")
    return dataz


def weekly_move_likelihood(ticker, delta):
    if delta == 0:
        return 1.0

    ticker_data = readStock(ticker)
    price_info = ticker_data[3000:, [1, 2, 3]]
    hts = Ticker(price_info, 5)

    def calculate_stats(price_data):
        price_delta = get_range_diff(price_data, 7)
        total = price_delta.shape[0]
        data_hist_prob, data_hist_delta = np.histogram(price_delta, 40)
        pos = np.absolute(data_hist_delta-delta).argmin()
        if delta > 0:
            res_prob = np.sum(data_hist_prob[pos+1:])/total
        else:
            res_prob = np.sum(data_hist_prob[:pos+1])/total
        return res_prob
    return round(calculate_stats(price_info[-2399:])*10000.0)/10000.0, round(calculate_stats(price_info[-399:])*10000.0)/10000.0


if __name__ == '__main__':
    # get30daydance("AAPL")
    # print(uniq)
    # datas = get30daydelta("GOOG")
    ticker_data = readStock("AMZN")
    price_info = ticker_data[-3000:, 4]

    icondor = IronCondor(115.0, 123.0, 125.0, 133.0, 4.22, 7.14, 7.86, 4.43, 53)
    icondor.calculate_domain(121.14, 0.03, 0.42)
    icondor.get_breakevens()

    # hts = Ticker(price_info, 20)

    # fff = hts.calculate_envelopes()
    # long_x, near_x = hts.calc_probability(0.05)
    # print(long_x.high)
    #
    # datas_close = [long_x.close, near_x.close]
    # datas_high = [long_x.high, near_x.high]
    # datas_low = [long_x.low, near_x.low]
    # pyplot.bar([0, 0.5], datas_high, 0.1, color='r', label='High', edgecolor='gray')
    # pyplot.bar([0.1, 0.6], datas_close, 0.1, color='b', label='Close', edgecolor='gray')
    # pyplot.bar([0.2, 0.7], datas_low, 0.1, color='g', label='Low', edgecolor='gray')
    # pyplot.xticks([r/2 + 0.075 for r in range(2)],
    #            ['Long', 'Near'])
    # pyplot.legend()
    # pyplot.show()



    # move_delta = 0.1
    # ticker_sym = "MSFT"
    # weekly_prob_pos = weekly_move_likelihood(ticker_sym, 0.05)
    # weekly_prob_neg = weekly_move_likelihood(ticker_sym, -0.06)
    # print(f'{(weekly_prob_neg[0])*100.0}%/{(weekly_prob_neg[1])*100.0}%')
    # datas = get30daydance("MSFT")
    # pyplot.hist(datas, bins=30)
    # pyplot.show()
