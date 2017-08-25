from logging import ERROR
from typing import Iterable, Optional

from lambdatrader.config import (
    RETRACEMENT_SIGNALS__ORDER_TIMEOUT,
    RETRACEMENT_SIGNALS__HIGH_VOLUME_LIMIT,
    RETRACEMENT_SIGNALS__BUY_PROFIT_FACTOR,
    RETRACEMENT_SIGNALS__RETRACEMENT_RATIO,
)
from lambdatrader.models.tradesignal import (
    PriceEntry, PriceTakeProfitSuccessExit, TimeoutStopLossFailureExit, TradeSignal,
)
from loghandlers import get_logger_with_all_handlers


class BaseSignalGenerator:

    def __init__(self, market_info, live=False, silent=False):
        self.market_info = market_info
        self.LIVE = live
        self.SILENT = silent

        self.logger = get_logger_with_all_handlers(__name__)

        if not self.LIVE:
            self.logger.setLevel(ERROR)

    def generate_signals(self):
        self.debug('generate_signals')
        trade_signals = self.__analyze_market()
        return trade_signals

    def __analyze_market(self):
        self.debug('__analyze_market')
        high_volume_pairs = self.get_allowed_pairs()
        trade_signals = list(self.__analyze_pairs(pairs=high_volume_pairs))
        return trade_signals

    def __analyze_pairs(self, pairs) -> Iterable[TradeSignal]:
        self.debug('__analyze_pairs')
        for pair in pairs:
            trade_signal = self.analyze_pair(pair=pair)
            if trade_signal:
                yield trade_signal

    def get_allowed_pairs(self):
        raise NotImplementedError

    def analyze_pair(self, pair):
        raise NotImplementedError

    def get_market_date(self):
        return self.market_info.get_market_date()

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, args, kwargs=kwargs)

    def backtest_print(self, *args):
        if not self.LIVE and not self.SILENT:
            print(args)


class RetracementSignalGenerator(BaseSignalGenerator):

    HIGH_VOLUME_LIMIT = RETRACEMENT_SIGNALS__HIGH_VOLUME_LIMIT
    ORDER_TIMEOUT = RETRACEMENT_SIGNALS__ORDER_TIMEOUT
    BUY_PROFIT_FACTOR = RETRACEMENT_SIGNALS__BUY_PROFIT_FACTOR
    RETRACEMENT_RATIO = RETRACEMENT_SIGNALS__RETRACEMENT_RATIO

    def get_allowed_pairs(self):
        self.debug('get_allowed_pairs')
        high_volume_pairs = self.__get_high_volume_pairs()
        self.debug('high_volume_pairs:%s', str(high_volume_pairs))
        return high_volume_pairs

    def analyze_pair(self, pair) -> Optional[TradeSignal]:
        self.debug('analyze_pair')

        latest_ticker = self.market_info.get_pair_ticker(pair=pair)
        price = latest_ticker.lowest_ask
        market_date = self.get_market_date()

        self.debug('market_date:%s', str(market_date))
        self.debug('latest_ticker:%s', str(latest_ticker))

        target_price = price * self.BUY_PROFIT_FACTOR
        day_high_price = latest_ticker.high24h

        self.debug('target_price:%s', str(target_price))
        self.debug('day_high_price:%s', str(day_high_price))

        price_is_lower_than_day_high = target_price < day_high_price

        self.debug('price_is_lower_than_day_high:%s', str(price_is_lower_than_day_high))

        if not price_is_lower_than_day_high:
            return

        current_retracement_ratio = (target_price - price) / (day_high_price - price)
        retracement_ratio_satisfied = current_retracement_ratio <= self.RETRACEMENT_RATIO

        self.debug('current_retracement_ratio:%s', str(current_retracement_ratio))
        self.debug('retracement_ratio_satisfied:%s', str(retracement_ratio_satisfied))

        if retracement_ratio_satisfied:
            entry = PriceEntry(price)
            success_exit = PriceTakeProfitSuccessExit(price=target_price)
            failure_exit = TimeoutStopLossFailureExit(timeout=self.ORDER_TIMEOUT)

            trade_signal = TradeSignal(date=market_date, exchange=None, pair=pair, entry=entry,
                                       success_exit=success_exit, failure_exit=failure_exit)

            self.logger.info('trade_signal:%s', str(trade_signal))

            return trade_signal

    def __get_high_volume_pairs(self):
        self.debug('__get_high_volume_pairs')
        return sorted(
            filter(lambda p: self.market_info.get_pair_last_24h_btc_volume(p) >= self.HIGH_VOLUME_LIMIT,
                   self.market_info.get_active_pairs()),
            key=lambda pair: -self.market_info.get_pair_last_24h_btc_volume(pair=pair)
        )
