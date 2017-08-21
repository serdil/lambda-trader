class Candlestick:

    def __init__(self, date, high, low, _open, close, base_volume, quote_volume, weighted_average):
        self.date = date
        self.high = high
        self.low = low
        self.open = _open
        self.close = close
        self.base_volume = base_volume
        self.quote_volume = quote_volume
        self.weighted_average = weighted_average
