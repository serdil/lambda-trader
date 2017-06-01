
class MarketInfo:
    def __init__(self, pairinfos):
        self.pairs = {}
        for pairinfo in pairinfos:
            self.pairs[pairinfo.currency_pair] = pairinfo