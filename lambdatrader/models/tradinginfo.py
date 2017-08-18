class TradingInfo:
    def __init__(self, history_start, history_end, estimated_balances, frozen_balances, trades):
        self.history_start = history_start
        self.history_end = history_end
        self.estimated_balances = estimated_balances
        self.frozen_balances = frozen_balances
        self.trades = trades

    def __repr__(self):
        return '''TradingInfo(
        history_start: {}
        history_end: {}
        balances: {}
        frozen_balances: {}
        trades: {})'''.format(self.history_start, self.history_end,
                              self.estimated_balances, self.frozen_balances, self.trades)
