from typing import Iterable

from lambdatrader.loghandlers import (
    get_trading_logger,
)
from lambdatrader.models.tradesignal import (
    TradeSignal,
)
from lambdatrader.signals.analysis import Analysis


class BaseSignalGenerator:

    def __init__(self, market_info, live=False, silent=False, **kwargs):
        self.market_info = market_info
        self.LIVE = live
        self.SILENT = silent

        self.logger = get_trading_logger(__name__, live=live, silent=silent)

        self.analysis = Analysis(market_info=market_info, live=live, silent=silent)

    def generate_signals(self, tracked_signals):
        self.debug('generate_signals')
        trade_signals = self.analyze_market(tracked_signals=tracked_signals)
        return trade_signals

    def analyze_market(self, tracked_signals):
        self.pre_analyze_market(tracked_signals)
        self.debug('analyze_market')
        allowed_pairs = self.get_allowed_pairs()
        self.market_info.fetch_ticker()
        trade_signals = list(self.analyze_pairs(pairs=allowed_pairs,
                                                tracked_signals=tracked_signals))
        self.post_analyze_market(tracked_signals)
        return trade_signals

    def analyze_pairs(self, pairs, tracked_signals) -> Iterable[TradeSignal]:
        self.debug('analyze_pairs')
        for pair in pairs:
            trade_signal = self.analyze_pair(pair=pair, tracked_signals=tracked_signals)
            if trade_signal:
                yield trade_signal

    def get_allowed_pairs(self):
        raise NotImplementedError

    def analyze_pair(self, pair, tracked_signals):
        raise NotImplementedError

    def get_market_date(self):
        return self.market_date

    @property
    def market_date(self):
        return self.market_info.market_date

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def backtest_print(self, *args):
        if not self.LIVE and not self.SILENT:
            print(*args)

    def pre_analyze_market(self, tracked_signals):
        pass

    def post_analyze_market(self, tracked_signals):
        pass


