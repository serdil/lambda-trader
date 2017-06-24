class Candlestick:

    def __init__(
            self, open, close, high, low, base_volume, quote_volume, timestamp,
            weighted_average=None
    ):
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.base_volume = base_volume
        self.quote_volume = quote_volume
        self.timestamp = timestamp
        self.weighted_average = weighted_average
