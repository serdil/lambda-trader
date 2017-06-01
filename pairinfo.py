
class PairInfo:
    def __init__(self, pair_name, candlesticks):
        self.pair_name = pair_name
        self.history = []
        for candlestick in candlesticks:
            self.history.append(candlestick)
        self.history.sort(key=lambda c: c.timestamp)