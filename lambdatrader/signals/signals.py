from typing import Iterable, Optional

from lambdatrader.config import (
    RETRACEMENT_SIGNALS__ORDER_TIMEOUT,
    RETRACEMENT_SIGNALS__HIGH_VOLUME_LIMIT,
    RETRACEMENT_SIGNALS__BUY_PROFIT_FACTOR,
    RETRACEMENT_SIGNALS__RETRACEMENT_RATIO,
)
from models.tradesignal import (
    PriceEntry, PriceTakeProfitSuccessExit, TimeoutStopLossFailureExit, TradeSignal,
)


class SignalGenerator:

    HIGH_VOLUME_LIMIT = RETRACEMENT_SIGNALS__HIGH_VOLUME_LIMIT
    ORDER_TIMEOUT = RETRACEMENT_SIGNALS__ORDER_TIMEOUT
    BUY_PROFIT_FACTOR = RETRACEMENT_SIGNALS__BUY_PROFIT_FACTOR
    RETRACEMENT_RATIO = RETRACEMENT_SIGNALS__RETRACEMENT_RATIO

    def __init__(self, market_info):
        self.market_info = market_info

    def generate_signals(self):
        trade_signals = self.__analyze_market()
        return trade_signals

    def __analyze_market(self):
        high_volume_pairs = self.__get_high_volume_pairs()
        trade_signals = list(self.__analyze_pairs(pairs=high_volume_pairs))
        return trade_signals

    def __analyze_pairs(self, pairs) -> Iterable[TradeSignal]:
        for pair in pairs:
            trade_signal = self.__analyze_pair(pair=pair)
            if trade_signal:
                yield trade_signal

    def __analyze_pair(self, pair) -> Optional[TradeSignal]:
        latest_ticker = self.market_info.get_pair_ticker(pair=pair)
        price = latest_ticker.lowest_ask
        market_date = self.__get_market_date()

        target_price = price * self.BUY_PROFIT_FACTOR
        day_high_price = latest_ticker.high24h

        price_is_lower_than_day_high = target_price < day_high_price

        if not price_is_lower_than_day_high:
            return

        current_retracement_ratio = (target_price - price) / (day_high_price - price)
        retracement_ratio_satisfied = current_retracement_ratio <= self.RETRACEMENT_RATIO

        if retracement_ratio_satisfied:
            entry = PriceEntry(price)
            success_exit = PriceTakeProfitSuccessExit(price=target_price)
            failure_exit = TimeoutStopLossFailureExit(timeout=self.ORDER_TIMEOUT)

            trade_signal = TradeSignal(date=market_date, exchange=None, pair=pair, entry=entry,
                                       success_exit=success_exit, failure_exit=failure_exit)

            return trade_signal

    def __get_high_volume_pairs(self):
        return sorted(
            filter(lambda p: self.market_info.get_pair_last_24h_btc_volume(p) >= self.HIGH_VOLUME_LIMIT,
                   self.market_info.pairs()),
            key=lambda pair: -self.market_info.get_pair_last_24h_btc_volume(pair=pair)
        )

    def __get_market_date(self):
        return self.market_info.get_market_time()
