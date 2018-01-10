from typing import Iterable, Optional

from lambdatrader.config import (
    RETRACEMENT_SIGNALS__ORDER_TIMEOUT, RETRACEMENT_SIGNALS__HIGH_VOLUME_LIMIT,
    RETRACEMENT_SIGNALS__BUY_PROFIT_FACTOR, RETRACEMENT_SIGNALS__RETRACEMENT_RATIO,
    DYNAMIC_RETRACEMENT_SIGNALS__LOOKBACK_DRAWDOWN_RATIO,
    DYNAMIC_RETRACEMENT_SIGNALS__LOOKBACK_DAYS, RED_GREEN_MARKET_NUM_PAIRS, RED_MARKET_MAJORITY_NUM,
    RED_MARKET_NUM_CANDLES, RED_MARKET_DIP_THRESHOLD,

    GREEN_MARKET_MAJORITY_NUM, GREEN_MARKET_NUM_CANDLES, GREEN_MARKET_UP_THRESHOLD,
    ENABLING_DISABLING_CHECK_INTERVAL,
)
from lambdatrader.loghandlers import (
    get_trading_logger,
)
from lambdatrader.models.tradesignal import (
    PriceEntry, PriceTakeProfitSuccessExit, TimeoutStopLossFailureExit, TradeSignal,
)
from lambdatrader.signals.analysis import Analysis
from lambdatrader.utilities.decorators import every_n_market_seconds
from lambdatrader.utilities.utils import seconds


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


class RetracementSignalGenerator(BaseSignalGenerator):

    HIGH_VOLUME_LIMIT = RETRACEMENT_SIGNALS__HIGH_VOLUME_LIMIT
    ORDER_TIMEOUT = RETRACEMENT_SIGNALS__ORDER_TIMEOUT
    BUY_PROFIT_FACTOR = RETRACEMENT_SIGNALS__BUY_PROFIT_FACTOR
    RETRACEMENT_RATIO = RETRACEMENT_SIGNALS__RETRACEMENT_RATIO

    def get_allowed_pairs(self):
        self.debug('get_allowed_pairs')
        high_volume_pairs = self.analysis.get_high_volume_pairs(
            high_volume_limit=self.HIGH_VOLUME_LIMIT
        )
        return high_volume_pairs

    def analyze_pair(self, pair, tracked_signals) -> Optional[TradeSignal]:

        if pair in [signal.pair for signal in tracked_signals]:
            self.debug('pair_already_in_tracked_signals:%s', pair)
            return

        latest_ticker = self.market_info.get_pair_ticker(pair=pair)
        price = latest_ticker.lowest_ask
        market_date = self.market_date

        target_price = price * self.BUY_PROFIT_FACTOR
        day_high_price = latest_ticker.high24h

        price_is_lower_than_day_high = target_price < day_high_price

        if not price_is_lower_than_day_high:
            return

        current_retracement_ratio = (target_price - price) / (day_high_price - price)
        retracement_ratio_satisfied = current_retracement_ratio <= self.RETRACEMENT_RATIO

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

            self.logger.debug('trade_signal:%s', str(trade_signal))

            return trade_signal


class DynamicRetracementSignalGenerator(BaseSignalGenerator):  # TODO deduplicate logic

    NUM_TRADING_PAIRS = 50
    HIGH_VOLUME_LIMIT = RETRACEMENT_SIGNALS__HIGH_VOLUME_LIMIT
    BUY_PROFIT_FACTOR = RETRACEMENT_SIGNALS__BUY_PROFIT_FACTOR

    LOOKBACK_DRAWDOWN_RATIO = DYNAMIC_RETRACEMENT_SIGNALS__LOOKBACK_DRAWDOWN_RATIO
    LOOKBACK_DAYS = DYNAMIC_RETRACEMENT_SIGNALS__LOOKBACK_DAYS

    RED_GREEN_MARKET_NUM_PAIRS = RED_GREEN_MARKET_NUM_PAIRS

    RED_MARKET_MAJORITY_NUM = RED_MARKET_MAJORITY_NUM
    RED_MARKET_NUM_CANDLES = RED_MARKET_NUM_CANDLES
    RED_MARKET_DIP_THRESHOLD = RED_MARKET_DIP_THRESHOLD

    GREEN_MARKET_MAJORITY_NUM = GREEN_MARKET_MAJORITY_NUM
    GREEN_MARKET_NUM_CANDLES = GREEN_MARKET_NUM_CANDLES
    GREEN_MARKET_UP_THRESHOLD = GREEN_MARKET_UP_THRESHOLD

    ENABLING_DISABLING_CHECK_INTERVAL = ENABLING_DISABLING_CHECK_INTERVAL

    def __init__(self, market_info, live=False, silent=False, enable_disable=True, **kwargs):
        super().__init__(market_info, live=live, silent=silent)
        self.pairs_retracement_ratios = {}
        self.enable_disable = enable_disable
        self.trading_enabled = True
        self.last_enable_disable_checked = 0

    def get_algo_descriptor(self):
        return {
            'CLASS_NAME': self.__class__.__name__,

            'NUM_TRADING_PAIRS': self.NUM_TRADING_PAIRS,
            'HIGH_VOLUME_LIMIT': self.HIGH_VOLUME_LIMIT,
            'BUY_PROFIT_FACTOR': self.BUY_PROFIT_FACTOR,

            'LOOKBACK_DRAWDOWN_RATIO': self.LOOKBACK_DRAWDOWN_RATIO,
            'LOOKBACK_DAYS': self.LOOKBACK_DAYS,

            'RED_GREEN_MARKET_NUM_PAIRS': self.RED_GREEN_MARKET_NUM_PAIRS,

            'RED_MARKET_MAJORITY_NUM': self.RED_MARKET_MAJORITY_NUM,
            'RED_MARKET_NUM_CANDLES': self.RED_MARKET_NUM_CANDLES,
            'RED_MARKET_DIP_THRESHOLD': self.RED_MARKET_DIP_THRESHOLD,

            'GREEN_MARKET_MAJORITY_NUM': self.GREEN_MARKET_MAJORITY_NUM,
            'GREEN_MARKET_NUM_CANDLES': self.GREEN_MARKET_NUM_CANDLES,
            'GREEN_MARKET_UP_THRESHOLD': self.GREEN_MARKET_UP_THRESHOLD,
        }

    def get_allowed_pairs(self):
        self.debug('get_allowed_pairs')
        high_volume_pairs = [pair for pair
                             in self.analysis.get_high_volume_pairs(self.HIGH_VOLUME_LIMIT)
                             if pair not in ['BTC_DOGE', 'BTC_BCN']]
        high_volume_pairs = high_volume_pairs[:self.NUM_TRADING_PAIRS]
        return high_volume_pairs

    def pre_analyze_market(self, tracked_signals):
        if self.enable_disable:
            self.update_enabling_disabling_status(tracked_signals)

    @every_n_market_seconds(n=ENABLING_DISABLING_CHECK_INTERVAL)
    def update_enabling_disabling_status(self, tracked_signals):
        if self.trading_enabled:
            should_disable = self.should_disable_trading(tracked_signals=tracked_signals)
            if should_disable:
                self.logger.info('disabling_trading')
                self.backtest_print('++++++++++++++++++++++++'
                                    'DISABLING_TRADING'
                                    '++++++++++++++++++++++++')
                self.trading_enabled = False
                self.cancel_all_trades(tracked_signals)
        else:
            should_enable = self.should_enable_trading(tracked_signals=tracked_signals)
            if should_enable:
                self.logger.info('enabling_trading')
                self.backtest_print('========================'
                                    'ENABLING_TRADING'
                                    '========================')
                self.trading_enabled = True

    @staticmethod
    def cancel_all_trades(tracked_signals):
        for signal in tracked_signals:
            signal.cancel()

    def analyze_pair(self, pair, tracked_signals) -> Optional[TradeSignal]:

        if not self.trading_enabled:
            return

        try:
            self.pairs_retracement_ratios[pair] = \
                self.analysis.calc_pair_retracement_ratio(
                    pair,
                    buy_profit_factor=self.BUY_PROFIT_FACTOR,
                    lookback_days=self.LOOKBACK_DAYS,
                    lookback_drawdown_ratio=self.LOOKBACK_DRAWDOWN_RATIO
                )
        except KeyError:
            self.logger.warning('Key error while getting candlestick for pair: %s', pair)
            return

        if pair in [signal.pair for signal in tracked_signals]:
            self.debug('pair_already_in_tracked_signals:%s', pair)
            return

        latest_ticker = self.market_info.get_pair_ticker(pair=pair)
        price = latest_ticker.lowest_ask
        market_date = self.market_date

        target_price = price * self.BUY_PROFIT_FACTOR
        period_high_price = self.analysis.calc_n_days_high(pair=pair, num_days=self.LOOKBACK_DAYS)

        price_is_lower_than_period_high = target_price < period_high_price

        if not price_is_lower_than_period_high:
            return

        current_retracement_ratio = (target_price - price) / (period_high_price - price)
        retracement_ratio_satisfied = (current_retracement_ratio <=
                                       self.pairs_retracement_ratios[pair])

        if retracement_ratio_satisfied:
            self.debug('retracement_ratio_satisfied')
            self.debug('current_retracement_ratio:%s', str(current_retracement_ratio))
            self.debug('market_date:%s', str(market_date))
            self.debug('latest_ticker:%s:%s', pair, str(latest_ticker))
            self.debug('target_price:%s', str(target_price))
            self.debug('day_high_price:%s', str(period_high_price))

            entry = PriceEntry(price)
            success_exit = PriceTakeProfitSuccessExit(price=target_price)
            failure_exit = TimeoutStopLossFailureExit(timeout=seconds(days=self.LOOKBACK_DAYS))

            trade_signal = TradeSignal(date=market_date, exchange=None, pair=pair, entry=entry,
                                       success_exit=success_exit, failure_exit=failure_exit)

            self.logger.debug('trade_signal:%s', str(trade_signal))

            return trade_signal

    def should_disable_trading(self, tracked_signals):
        return self.analysis.should_stop_trading_based_on_market_red(
            num_pairs=self.RED_GREEN_MARKET_NUM_PAIRS,
            majority_num=self.RED_MARKET_MAJORITY_NUM,
            num_candles=self.RED_MARKET_NUM_CANDLES,
            dip_threshold=self.RED_MARKET_DIP_THRESHOLD
        )

    def should_enable_trading(self, tracked_signals):
        return self.analysis.should_start_trading_based_on_market_green(
            num_pairs=self.RED_GREEN_MARKET_NUM_PAIRS,
            majority_num=self.GREEN_MARKET_MAJORITY_NUM,
            num_candles=self.GREEN_MARKET_NUM_CANDLES,
            up_threshold=self.GREEN_MARKET_UP_THRESHOLD
        )
