import blist

class PairInfo:
    def __init__(self, currency_pair, candlesticks=[]):
        self.currency_pair = currency_pair
        self.history = blist.sorteddict()
        for candlestick in candlesticks:
            self.add_candlestick(candlestick)

    def add_candlestick(self, candlestick):
        self.history[candlestick.timestamp] = candlestick

    def get_candlestick(self, timestamp):
        return self.history[timestamp]

    def get_start_time(self):
        return self.history[self.history.keys()[0]].timestamp

    def get_end_time(self):
        return self.history[self.history.keys()[-1]].timestamp
