
class PairInfo:
    def __init__(self, currency_pair, candlesticks):
        self.currency_pair = currency_pair
        self.history = []
        for candlestick in candlesticks:
            self.history.append(candlestick)
        self.history.sort(key=lambda c: c.timestamp)