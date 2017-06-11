
class Ticker:
    def __init__(self, lowest_ask, highest_bid, last, base_volume, quote_volume, percent_change, low24h, high24h, is_frozen=False, id=None):
        self.is_frozen = is_frozen
        self.id = id
        self.lowest_ask = lowest_ask
        self.highest_bid = highest_bid
        self.last = last
        self.base_volume = base_volume
        self.quote_volume = quote_volume
        self.percent_change = percent_change
        self.low24h = low24h
        self.high24h = high24h