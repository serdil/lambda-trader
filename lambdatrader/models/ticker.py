
class Ticker:
    def __init__(self, lowest_ask, highest_bid, last, base_volume,
                 quote_volume, percent_change, low24h, high24h, is_frozen=False, _id=None):
        self.is_frozen = is_frozen
        self.id = _id
        self.lowest_ask = lowest_ask
        self.highest_bid = highest_bid
        self.last = last
        self.base_volume = base_volume
        self.quote_volume = quote_volume
        self.percent_change = percent_change
        self.low24h = low24h
        self.high24h = high24h

    def __repr__(self):
        return 'Ticker(base_volume=' + str(self.base_volume) + \
               ' lowest_ask=' + str(self.lowest_ask) + \
               ' highest_bid=' + str(self.highest_bid) + \
               ' last=' + str(self.last) + \
               ' high24h=' + str(self.high24h) + \
               ' low24h=' + str(self.low24h) + ')'
