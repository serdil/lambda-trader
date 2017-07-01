from models.trade import Trade
from models.tradinginfo import TradingInfo


class BaseStrategy:
    def __init__(self):
        self.__trade_starts = {}
        self.__trades = []
        self.__history_start = None
        self.__history_end = None
        self.__balances = {}

    def set_history_start(self, date):
        self.__history_start = date

    def set_history_end(self, date):
        self.__history_end = date

    def declare_trade_start(self, date, trade_id):
        self.__trade_starts[trade_id] = date

    def declare_trade_end(self, date, trade_id, profit_amount):
        start_date = self.__trade_starts[trade_id]
        end_date = date
        trade = Trade(id=trade_id, start_date=start_date, end_date=end_date, profit=profit_amount)
        self.__trades.append(trade)

    def declare_balance(self, date, balance):
        self.__balances[date] = balance

    def get_trading_info(self):
        return TradingInfo(history_start=self.__history_start, history_end=self.__history_end,
                           balances=dict(self.__balances), trades=list(self.__trades))
