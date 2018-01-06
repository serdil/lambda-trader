from logging import ERROR
from operator import itemgetter
from typing import Iterable, Optional

from lambdatrader.backtesting import backtest
from lambdatrader.backtesting.account import BacktestingAccount
from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestickstore import CandlestickStore
from lambdatrader.evaluation.utils import period_statistics
from lambdatrader.executors.executors import SignalExecutor
from lambdatrader.config import (
    RETRACEMENT_SIGNALS__ORDER_TIMEOUT, RETRACEMENT_SIGNALS__HIGH_VOLUME_LIMIT,
    RETRACEMENT_SIGNALS__BUY_PROFIT_FACTOR, RETRACEMENT_SIGNALS__RETRACEMENT_RATIO,
    DYNAMIC_RETRACEMENT_SIGNALS__LOOKBACK_DRAWDOWN_RATIO,
    DYNAMIC_RETRACEMENT_SIGNALS__LOOKBACK_DAYS,
)
from lambdatrader.constants import M5
from lambdatrader.loghandlers import (
    get_logger_with_all_handlers, get_logger_with_console_handler, get_silent_logger,
)
from lambdatrader.models.tradesignal import (
    PriceEntry, PriceTakeProfitSuccessExit, TimeoutStopLossFailureExit, TradeSignal,
)
from lambdatrader.utilities.utils import seconds


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
        self.pre_analyze_market(tracked_signals)
        self.debug('__analyze_market')
        allowed_pairs = self.get_allowed_pairs()
        self.market_info.fetch_ticker()
        trade_signals = list(self.__analyze_pairs(pairs=allowed_pairs, tracked_signals=tracked_signals))
        self.post_analyze_market(tracked_signals)
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
        return self.market_date

    @property
    def market_date(self):
        return self.market_info.market_date

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def backtest_print(self, *args):
        if not self.LIVE and not self.SILENT:
            print(args)

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
        high_volume_pairs = self.__get_high_volume_pairs()
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

    def __get_high_volume_pairs(self):
        self.debug('__get_high_volume_pairs')
        return sorted(
            filter(lambda p: self.market_info.get_pair_last_24h_btc_volume(p) >= self.HIGH_VOLUME_LIMIT,
                   self.market_info.get_active_pairs()),
            key=lambda pair: -self.market_info.get_pair_last_24h_btc_volume(pair=pair)
        )


class DynamicRetracementSignalGenerator(BaseSignalGenerator):  # TODO deduplicate logic

    HIGH_VOLUME_LIMIT = RETRACEMENT_SIGNALS__HIGH_VOLUME_LIMIT
    ORDER_TIMEOUT = RETRACEMENT_SIGNALS__ORDER_TIMEOUT
    BUY_PROFIT_FACTOR = 1.03

    LOOKBACK_DRAWDOWN_RATIO = DYNAMIC_RETRACEMENT_SIGNALS__LOOKBACK_DRAWDOWN_RATIO
    LOOKBACK_DAYS = DYNAMIC_RETRACEMENT_SIGNALS__LOOKBACK_DAYS

    DISABLING_BACKTESTING_TIME = seconds(hours=24)
    DISABLING_ROI_THRESHOLD = -0.05
    DISABLING_ROI_THRESHOLD_TIME = seconds(hours=2)

    ENABLING_BACKTESTING_TIME = seconds(hours=1)
    ENABLING_ROI_THRESHOLD = 0.03
    ENABLING_ROI_THRESHOLD_TIME = seconds(hours=3)

    RED_GREEN_MARKET_NUM_PAIRS = 50

    RED_MARKET_MAJORITY_NUM = 19
    RED_MARKET_NUM_CANDLES = 2
    RED_MARKET_DIP_THRESHOLD = 0.01

    GREEN_MARKET_MAJORITY_NUM = 25
    GREEN_MARKET_NUM_CANDLES = 6
    GREEN_MARKET_UP_THRESHOLD = 0.00

    ENABLING_DISABLING_CHECK_INTERVAL = seconds(minutes=0)

    def __init__(self, market_info, live=False, silent=False, enable_disable=True):
        super().__init__(market_info, live=live, silent=silent)
        self.pairs_retracement_ratios = {}
        self.enable_disable = enable_disable
        self.trading_enabled = True
        self.last_enable_disable_checked = 10

    def get_allowed_pairs(self):
        self.debug('get_allowed_pairs')
        high_volume_pairs = [pair for pair in self.__get_high_volume_pairs()
                             if pair not in ['BTC_DOGE', 'BTC_BCN']]
        return high_volume_pairs

    def pre_analyze_market(self, tracked_signals):
        if self.enable_disable:
            market_date = self.market_info.get_market_date()
            time_since_last_check = market_date - self.last_enable_disable_checked
            if time_since_last_check >= self.ENABLING_DISABLING_CHECK_INTERVAL:
                self.update_enabling_disabling_status(tracked_signals)

    def update_enabling_disabling_status(self, tracked_signals):
        # print('updating enabling disabling status')
        self.last_enable_disable_checked = self.market_info.get_market_date()
        if self.trading_enabled:
            should_disable = self.should_disable_trading(tracked_signals)
            if should_disable:
                print('====================DISABLING TRADING==========================')
                self.trading_enabled = False
                self.cancel_all_trades(tracked_signals)
        else:
            should_enable = self.should_enable_trading()
            if should_enable:
                print('++++++++++++++++++++ENABLING TRADING+++++++++++++++++++++++++++')
                self.trading_enabled = True

    def cancel_all_trades(self, tracked_signals):
        for signal in tracked_signals:
            signal.cancel()

    def analyze_pair(self, pair, tracked_signals) -> Optional[TradeSignal]:

        if not self.trading_enabled:
            return

        try:
            self.pairs_retracement_ratios[pair] = self.__calc_pair_retracement_ratio(pair)
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
        period_high_price = self.__calc_n_days_high(pair=pair, num_days=self.LOOKBACK_DAYS)

        price_is_lower_than_period_high = target_price < period_high_price

        if not price_is_lower_than_period_high:
            return

        current_retracement_ratio = (target_price - price) / (period_high_price - price)
        retracement_ratio_satisfied = current_retracement_ratio <= \
                                      self.pairs_retracement_ratios[pair]

        if retracement_ratio_satisfied:
            self.debug('retracement_ratio_satisfied')
            self.debug('current_retracement_ratio:%s', str(current_retracement_ratio))
            self.debug('market_date:%s', str(market_date))
            self.debug('latest_ticker:%s:%s', pair, str(latest_ticker))
            self.debug('target_price:%s', str(target_price))
            self.debug('day_high_price:%s', str(period_high_price))

            entry = PriceEntry(price)
            success_exit = PriceTakeProfitSuccessExit(price=target_price)
            failure_exit = TimeoutStopLossFailureExit(timeout=
                                                      self.days_to_seconds(self.LOOKBACK_DAYS))

            trade_signal = TradeSignal(date=market_date, exchange=None, pair=pair, entry=entry,
                                       success_exit=success_exit, failure_exit=failure_exit)

            self.logger.debug('trade_signal:%s', str(trade_signal))

            return trade_signal

    def __calc_pair_retracement_ratio(self, pair):
        period_max_drawdown = self.__calc_max_drawdown_since_n_days(pair, self.LOOKBACK_DAYS)

        if period_max_drawdown == 0:
            return 0.000000001

        return (self.BUY_PROFIT_FACTOR-1) / period_max_drawdown / self.LOOKBACK_DRAWDOWN_RATIO

    def __calc_max_drawdown_since_n_days(self, pair, num_days):
        lookback_num_candles = self.days_to_candlesticks(num_days)
        cur_max = -1
        min_since_cur_max = 1000000

        max_drawdown_max = 0
        max_drawdown_range = 0

        for i in range(lookback_num_candles - 1, -1, -1):
            candle = self.market_info.get_pair_candlestick(pair, i)

            if candle.high > cur_max:
                cur_max = candle.high
                min_since_cur_max = candle.low

            if candle.low < min_since_cur_max:
                min_since_cur_max = candle.low

            if cur_max - min_since_cur_max > max_drawdown_range:
                max_drawdown_max = cur_max
                max_drawdown_range = cur_max - min_since_cur_max

        if max_drawdown_max == 0:
            return 0

        period_max_drawdown = max_drawdown_range / max_drawdown_max
        return period_max_drawdown

    def __calc_n_days_high(self, pair, num_days):
        lookback_num_candles = self.days_to_candlesticks(num_days)
        high = 0

        for i in range(lookback_num_candles - 1, -1, -1):
            candle = self.market_info.get_pair_candlestick(pair, i)

            if candle.high > high:
                high = candle.high

        return high

    def should_disable_trading(self, tracked_signals):
        return self.should_stop_trading_based_on_market_red()

    def should_stop_trading_based_on_recent_constant_drawdown(self):
        trading_info = self.get_backtesting_trading_info(
            backtesting_time=self.DISABLING_BACKTESTING_TIME)

        estimated_balances_list = self.get_estimated_balances_list(trading_info)
        start_date = self.market_info.get_market_date() - self.DISABLING_ROI_THRESHOLD_TIME

        start_ind = self.find_smaller_equal_date_index(estimated_balances_list, start_date)

        max_balance = max([balance for date, balance in estimated_balances_list[:start_ind]])

        for date, balance in estimated_balances_list[start_ind:]:
            drawdown = (balance - max_balance) / max_balance
            if drawdown > self.DISABLING_ROI_THRESHOLD:
                return False

        return True

    def should_stop_trading_based_on_roi(self):
        trading_info = self.get_backtesting_trading_info(
            backtesting_time=self.DISABLING_BACKTESTING_TIME)
        stats = period_statistics(trading_info=trading_info)
        return stats['roi_live'] <= self.DISABLING_ROI_THRESHOLD

    def should_stop_trading_based_on_recent_roi(self):
        trading_info = self.get_backtesting_trading_info(
            backtesting_time=self.DISABLING_BACKTESTING_TIME)

        estimated_balances_list = self.get_estimated_balances_list(trading_info)
        start_date = self.market_info.get_market_date() - self.DISABLING_ROI_THRESHOLD_TIME

        start_ind = self.find_smaller_equal_date_index(estimated_balances_list, start_date)

        start_balance = estimated_balances_list[start_ind][1]
        end_balance = estimated_balances_list[len(estimated_balances_list)-1][1]

        recent_roi = (end_balance - start_balance) / start_balance
        return recent_roi <= self.DISABLING_ROI_THRESHOLD

    def should_stop_trading_based_on_adx_di(self):
        raise NotImplementedError

    def should_stop_trading_based_on_unseen_btc_price(self):
        lookback_num_candles = self.days_to_candlesticks(7)
        lowest = float('inf')
        highest = float('-inf')
        for i in range(lookback_num_candles+1, 0, -1):
            candle = self.market_info.get_pair_candlestick('USDT_BTC', ind=i)
            lowest = min(candle.low, lowest)
            highest = max(candle.high, highest)

        cur_candle = self.market_info.get_pair_candlestick('USDT_BTC', ind=0)
        return cur_candle.low < lowest or cur_candle.high > highest

    def should_stop_trading_based_on_majority_dip(self, tracked_signals):
        return self.majority_change(tracked_signals, -0.0075)

    def majority_change(self, tracked_signals, down_up_limit):
        dip_mode = down_up_limit < 0

        majority = 0.6
        candle_period = M5
        num_candles = 2

        total = 0
        num_dipped_upped = 0

        market_date = self.market_date

        for signal in tracked_signals:
            if market_date - signal.date > num_candles * candle_period.seconds():
                total += 1
                this_candle = self.market_info.get_pair_candlestick(signal.pair, ind=0,
                                                                    period=candle_period)
                old_candle = self.market_info.get_pair_candlestick(signal.pair, ind=num_candles,
                                                                   period=candle_period)
                old_price = old_candle.close
                this_price = this_candle.close

                if dip_mode:
                    if this_price < old_price:
                        this_dip = (this_price - old_price) / old_price
                        print(signal.pair, 'change:', this_dip)
                        if this_dip <= down_up_limit:
                            num_dipped_upped += 1
                else:
                    if this_price > old_price:
                        this_up = (this_price - old_price) / old_price
                        if this_up >= down_up_limit:
                            num_dipped_upped += 1

        if total > 0:
            print('num_dipped:', num_dipped_upped, 'total:', total)

        return total >= 5 and num_dipped_upped / total >= majority

    def should_stop_trading_based_on_market_red(self):
        return self.market_is_red(num_pairs=self.RED_GREEN_MARKET_NUM_PAIRS,
                                  majority_num=self.RED_MARKET_MAJORITY_NUM,
                                  num_candles=self.RED_MARKET_NUM_CANDLES,
                                  dip_threshold=self.RED_MARKET_DIP_THRESHOLD)

    #  50 tanenin yarisi son 15 dakikada en az %2 dusmus
    def market_is_red(self, num_pairs=50, majority_num=20, num_candles=3, dip_threshold=0.015):
        pairs = self.market_info.get_active_pairs()[:num_pairs]

        num_dipped = 0

        for pair in pairs:
            old_candle = self.market_info.get_pair_candlestick(pair, ind=num_candles)
            this_candle = self.market_info.get_pair_candlestick(pair, ind=0)

            old_price = old_candle.close
            this_price = this_candle.close

            if this_price < old_price:
                dip_amount = (old_price - this_price) / old_price
                if dip_amount >= dip_threshold:
                    num_dipped += 1

        print('num_dipped:', num_dipped)
        return num_dipped >= majority_num

    @staticmethod
    def get_estimated_balances_list(trading_info):
        estimated_balances_dict = trading_info.estimated_balances
        estimated_balances_list = sorted(estimated_balances_dict.items(), key=itemgetter(0))
        return estimated_balances_list

    @staticmethod
    def find_smaller_equal_date_index(estimated_balances_list, start_date):
        last_ind = 0
        for i, (date, balance) in enumerate(estimated_balances_list):
            if date > start_date:
                break
            last_ind = i
        return last_ind

    def should_enable_trading(self):
        return self.should_start_trading_based_on_market_green()

    def should_start_trading_based_on_roi(self):
        trading_info = self.get_backtesting_trading_info(
            backtesting_time=self.ENABLING_BACKTESTING_TIME)
        stats = period_statistics(trading_info=trading_info)
        return stats['roi_live'] >= self.ENABLING_ROI_THRESHOLD

    def should_start_trading_based_on_recent_roi(self):
        trading_info = self.get_backtesting_trading_info(
            backtesting_time=self.DISABLING_BACKTESTING_TIME)

        estimated_balances_list = self.get_estimated_balances_list(trading_info)
        start_date = self.market_info.get_market_date() - self.ENABLING_ROI_THRESHOLD_TIME

        start_ind = self.find_smaller_equal_date_index(estimated_balances_list, start_date)

        start_balance = estimated_balances_list[start_ind][1]
        end_balance = estimated_balances_list[len(estimated_balances_list) - 1][1]

        recent_roi = (end_balance - start_balance) / start_balance
        return recent_roi >= self.ENABLING_ROI_THRESHOLD

    @staticmethod
    def should_start_trading_unconditionally():
        return True

    def should_start_trading_based_on_market_green(self):
        return self.market_is_green(num_pairs=self.RED_GREEN_MARKET_NUM_PAIRS,
                                    majority_num=self.GREEN_MARKET_MAJORITY_NUM,
                                    num_candles=self.GREEN_MARKET_NUM_CANDLES,
                                    up_threshold=self.GREEN_MARKET_UP_THRESHOLD)

    def market_is_green(self, num_pairs=50, majority_num=25, num_candles=6, up_threshold=0.00):
        pairs = self.market_info.get_active_pairs()[:num_pairs]

        num_upped = 0

        for pair in pairs:
            old_candle = self.market_info.get_pair_candlestick(pair, ind=num_candles)
            this_candle = self.market_info.get_pair_candlestick(pair, ind=0)

            old_price = old_candle.close
            this_price = this_candle.close

            if this_price > old_price:
                up_amount = (this_price - old_price) / old_price
                if up_amount >= up_threshold:
                    num_upped += 1

        print('num_upped:', num_upped)
        return num_upped >= majority_num

    def get_backtesting_trading_info(self, backtesting_time):
        market_info = BacktestingMarketInfo(candlestick_store=CandlestickStore.get_instance())

        account = BacktestingAccount(market_info=market_info, balances={'BTC': 100})

        start_date = self.market_info.get_market_date() - backtesting_time
        end_date = self.market_info.get_market_date()

        signal_generators = [
            DynamicRetracementSignalGenerator(market_info=market_info,
                                              enable_disable=False, silent=True)
        ]
        signal_executor = SignalExecutor(market_info=market_info, account=account, silent=True)

        backtest.backtest(account=account, market_info=market_info,
                          signal_generators=signal_generators, signal_executor=signal_executor,
                          start=start_date, end=end_date, silent=True)

        return signal_executor.get_trading_info()

    @staticmethod
    def days_to_seconds(days):
        return int(days * 24 * 3600)

    @staticmethod
    def days_to_candlesticks(days, period=M5):
        return int(days * 24 * 3600 // period.seconds())

    def __get_high_volume_pairs(self):
        self.debug('__get_high_volume_pairs')
        return sorted(
            filter(lambda p: self.market_info.get_pair_last_24h_btc_volume(p) >=
                             self.HIGH_VOLUME_LIMIT,
                   self.market_info.get_active_pairs()),
            key=lambda pair: -self.market_info.get_pair_last_24h_btc_volume(pair=pair)
        )
