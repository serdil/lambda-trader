from lambdatrader.constants import M5, IRREGULAR


class Candlestick:

    def __init__(self, date, high, low, _open, close, base_volume,
                 quote_volume, weighted_average, period=M5, start_date=None):
        self.period = period

        if start_date is None:
            self.start_date = self.date - self.period.seconds()

        self.date = date
        self.high = high
        self.low = low
        self.open = _open
        self.close = close
        self.base_volume = base_volume
        self.quote_volume = quote_volume
        self.weighted_average = weighted_average

    def batch_with_candlesticks(self, candlesticks):
        batched_candlestick = Candlestick(self.date, self.high, self.low, self.open,
                                          self.close, self.base_volume, self.quote_volume,
                                          self.weighted_average)
        for candlestick in candlesticks:
            batched_candlestick = batched_candlestick.batch_with(candlestick)
        return batched_candlestick

    def batch_with(self, new_candlestick):
        start_date = self.start_date
        date = new_candlestick.date
        high = max(self.high, new_candlestick.high)
        low = min(self.low, new_candlestick.low)
        _open = self.open
        close = new_candlestick.close
        base_volume = self.base_volume + new_candlestick.base_volume
        quote_volume = self.quote_volume + new_candlestick.quote_volume

        this_ratio, that_ratio = self.normalize(self.date - self.start_date,
                                                new_candlestick.date - self.date)
        weighted_average = ((self.weighted_average * this_ratio +
                            new_candlestick.weighted_average * that_ratio) /
                            (this_ratio + that_ratio))

        return Candlestick(date, high, low, _open, close, base_volume,
                           quote_volume, weighted_average, period=IRREGULAR, start_date=start_date)

    @staticmethod
    def normalize(a, b):
        return a / max(a, b), b / max(a, b)

    @staticmethod
    def batch_candlesticks(candlesticks):
        if candlesticks:
            return candlesticks[0].batch_with_candlesticks(candlesticks[1:])

