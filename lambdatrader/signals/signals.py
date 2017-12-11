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
from lambdatrader.loghandlers import get_logger_with_all_handlers, get_logger_with_console_handler, get_silent_logger


class BaseSignalGenerator:

    def __init__(self, market_info, live=False, silent=False):
        self.market_info = market_info
        self.LIVE = live
        self.SILENT = silent

        if self.LIVE:
            self.logger = get_logger_with_all_handlers(__name__)
        elif self.SILENT:
            self.logger = get_silent_logger(__name__)
        else:
            self.logger = get_logger_with_console_handler(__name__)
            self.logger.setLevel(ERROR)

    def generate_signals(self, tracked_signals):
        self.debug('generate_signals')
        trade_signals = self.__analyze_market(tracked_signals=tracked_signals)
        return trade_signals

    def __analyze_market(self, tracked_signals):
        self.debug('__analyze_market')
        allowed_pairs = self.get_allowed_pairs()
        trade_signals = list(self.__analyze_pairs(pairs=allowed_pairs, tracked_signals=tracked_signals))
        return trade_signals

    def __analyze_pairs(self, pairs, tracked_signals) -> Iterable[TradeSignal]:
        self.debug('__analyze_pairs')
        for pair in pairs:
            trade_signal = self.analyze_pair(pair=pair, tracked_signals=tracked_signals)
            if trade_signal:
                yield trade_signal

    def get_allowed_pairs(self):
        raise NotImplementedError

    def analyze_pair(self, pair, tracked_signals):
        raise NotImplementedError

    def get_market_date(self):
        return self.market_info.get_market_date()

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def backtest_print(self, *args):
        if not self.LIVE and not self.SILENT:
            print(args)


class RetracementSignalGenerator(BaseSignalGenerator):

    HIGH_VOLUME_LIMIT = RETRACEMENT_SIGNALS__HIGH_VOLUME_LIMIT
    ORDER_TIMEOUT = RETRACEMENT_SIGNALS__ORDER_TIMEOUT
    BUY_PROFIT_FACTOR = RETRACEMENT_SIGNALS__BUY_PROFIT_FACTOR
    RETRACEMENT_RATIO = RETRACEMENT_SIGNALS__RETRACEMENT_RATIO

    WEEKLY_DRAWDOWN_RATIO = 0.5

    PAIRS_RETRACEMENT_RATIOS = {}

    def get_allowed_pairs(self):
        self.debug('get_allowed_pairs')
        high_volume_pairs = self.__get_high_volume_pairs()
        return high_volume_pairs

    def analyze_pair(self, pair, tracked_signals) -> Optional[TradeSignal]:

        self.PAIRS_RETRACEMENT_RATIOS[pair] = self.__calc_pair_retracement_ratio(pair)

        if pair in [signal.pair for signal in tracked_signals]:
            self.debug('pair_already_in_tracked_signals:%s', pair)
            return

        latest_ticker = self.market_info.get_pair_ticker(pair=pair)
        price = latest_ticker.lowest_ask
        market_date = self.get_market_date()

        target_price = price * self.BUY_PROFIT_FACTOR
        day_high_price = latest_ticker.high24h

        price_is_lower_than_day_high = target_price < day_high_price

        if not price_is_lower_than_day_high:
            return

        current_retracement_ratio = (target_price - price) / (day_high_price - price)
        retracement_ratio_satisfied = current_retracement_ratio <= \
                                      self.PAIRS_RETRACEMENT_RATIOS[pair]

        if retracement_ratio_satisfied:
            self.debug('retracement_ratio_satisfied')
            self.debug('current_retracement_ratio:%s', str(current_retracement_ratio))
            self.debug('market_date:%s', str(market_date))
            self.debug('latest_ticker:%s:%s', pair, str(latest_ticker))
            self.debug('target_price:%s', str(target_price))
            self.debug('day_high_price:%s', str(day_high_price))

            entry = PriceEntry(price)
            success_exit = PriceTakeProfitSuccessExit(price=target_price)
            failure_exit = TimeoutStopLossFailureExit(timeout=self.ORDER_TIMEOUT)

            trade_signal = TradeSignal(date=market_date, exchange=None, pair=pair, entry=entry,
                                       success_exit=success_exit, failure_exit=failure_exit)

            self.logger.info('trade_signal:%s', str(trade_signal))

            return trade_signal

    def __calc_pair_retracement_ratio(self, pair):
        one_week_num_candles = 24 * 3600 // 300
        cur_max = -1
        min_since_cur_max = 1000000

        max_drawdown_range = 0

        for i in range(one_week_num_candles-1, -1):
            candle = self.market_info.get_pair_candlestick(i)
            if candle.max > cur_max:
                cur_max = candle.max
                min_since_cur_max = candle.min

            if candle.min < min_since_cur_max:
                min_since_cur_max = candle.min

            if cur_max - min_since_cur_max > max_drawdown_range:
                max_drawdown_range = cur_max - min_since_cur_max

        return max_drawdown_range * self.WEEKLY_DRAWDOWN_RATIO



    def __get_high_volume_pairs(self):
        self.debug('__get_high_volume_pairs')
        return sorted(
            filter(lambda p: self.market_info.get_pair_last_24h_btc_volume(p) >= self.HIGH_VOLUME_LIMIT,
                   self.market_info.get_active_pairs()),
            key=lambda pair: -self.market_info.get_pair_last_24h_btc_volume(pair=pair)
        )
