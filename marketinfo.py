
class MarketInfo:
    def __init__(self, pairinfos):
        self.pairs = {}
        for pair_name, pairinfo in pairinfos.items():
            self.pairs[pair_name] = pairinfo