class TradingInfo:
    def __init__(self, history_start, history_end, balances, trades):
        self.history_start = history_start
        self.history_end = history_end
        self.balances = balances
        self.trades = trades

    def __repr__(self):
        return '''TradingInfo(
        history_start: {}
        history_end: {}
        balances: {}
        trades: {})'''.format(self.history_start, self.history_end, self.balances, self.trades)
