from lambdatrader.backtesting import backtest
from lambdatrader.backtesting.account import BacktestingAccount
from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestickstore import CandlestickStore
from lambdatrader.constants import M5
from lambdatrader.evaluation.utils import period_statistics
from lambdatrader.executors.executors import SignalExecutor
from lambdatrader.loghandlers import get_trading_logger
from lambdatrader.signals.analysis_utils import (
    get_estimated_balances_list, find_smaller_equal_date_index,
)
from lambdatrader.utilities.utils import candlesticks


class Analysis:

    def __init__(self, market_info, live=False, silent=False):
        self.market_info = market_info
        self.LIVE = live
        self.SILENT = silent
        self.logger = get_trading_logger(__name__, live=live, silent=silent)

    def backtest_print(self, *args):
        if not self.LIVE and not self.SILENT:
            print(*args)

    @property
    def market_date(self):
        return self.market_info.market_date

    def calc_pair_retracement_ratio(self,
                                    pair,
                                    lookback_days,
                                    buy_profit_factor,
                                    lookback_drawdown_ratio):
        period_max_drawdown = self.calc_max_drawdown_since_n_days(pair, lookback_days)

        if period_max_drawdown == 0:
            return 0.000000001

        return (buy_profit_factor-1) / period_max_drawdown / lookback_drawdown_ratio

    def calc_max_drawdown_since_n_days(self, pair, num_days):
        lookback_num_candles = candlesticks(days=num_days)
        cur_max = float('-inf')
        min_since_cur_max = float('inf')

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

    def calc_n_days_high(self, pair, num_days):
        lookback_num_candles = candlesticks(days=num_days)
        high = 0

        for i in range(lookback_num_candles - 1, -1, -1):
            candle = self.market_info.get_pair_candlestick(pair, i)

            if candle.high > high:
                high = candle.high

        return high

    def should_stop_trading_based_on_recent_constant_drawdown(self,
                                                              backtesting_time,
                                                              drawdown_threshold_time,
                                                              drawdown_threshold,
                                                              signal_generator_class):
        trading_info = self.get_backtesting_trading_info(
            backtesting_time=backtesting_time, signal_generator_class=signal_generator_class
        )

        estimated_balances_list = get_estimated_balances_list(trading_info)
        start_date = self.market_date - drawdown_threshold_time

        start_ind = find_smaller_equal_date_index(estimated_balances_list, start_date)

        max_balance = max([balance for date, balance in estimated_balances_list[:start_ind]])

        for date, balance in estimated_balances_list[start_ind:]:
            drawdown = (balance - max_balance) / max_balance
            if drawdown > drawdown_threshold:
                return False

        return True

    def should_stop_trading_based_on_roi(self,
                                         backtesting_time,
                                         roi_threshold,
                                         signal_generator_class):
        trading_info = self.get_backtesting_trading_info(
            backtesting_time=backtesting_time, signal_generator_class=signal_generator_class
        )
        stats = period_statistics(trading_info=trading_info)
        return stats['roi_live'] <= roi_threshold

    def should_stop_trading_based_on_recent_roi(self,
                                                backtesting_time,
                                                recent_roi_time,
                                                recent_roi_threshold,
                                                signal_generator_class):
        trading_info = self.get_backtesting_trading_info(
            backtesting_time=backtesting_time, signal_generator_class=signal_generator_class
        )

        estimated_balances_list = get_estimated_balances_list(trading_info)
        start_date = self.market_date - recent_roi_time

        start_ind = find_smaller_equal_date_index(estimated_balances_list, start_date)

        start_balance = estimated_balances_list[start_ind][1]
        end_balance = estimated_balances_list[len(estimated_balances_list)-1][1]

        recent_roi = (end_balance - start_balance) / start_balance
        return recent_roi <= recent_roi_threshold

    def should_stop_trading_based_on_adx_di(self):
        raise NotImplementedError

    def should_stop_trading_based_on_unseen_btc_price(self):
        lookback_num_candles = candlesticks(days=7)
        lowest = float('inf')
        highest = float('-inf')
        for i in range(lookback_num_candles+1, 0, -1):
            candle = self.market_info.get_pair_candlestick('USDT_BTC', ind=i)
            lowest = min(candle.low, lowest)
            highest = max(candle.high, highest)

        cur_candle = self.market_info.get_pair_candlestick('USDT_BTC', ind=0)
        return cur_candle.low < lowest or cur_candle.high > highest

    def should_stop_trading_based_on_majority_dip(self, tracked_signals):
        return self.majority_change_in_signals(tracked_signals, -0.0075)

    def majority_change_in_signals(self, tracked_signals, down_up_limit):
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
                        if this_dip <= down_up_limit:
                            num_dipped_upped += 1
                else:
                    if this_price > old_price:
                        this_up = (this_price - old_price) / old_price
                        if this_up >= down_up_limit:
                            num_dipped_upped += 1

        return total >= 5 and num_dipped_upped / total >= majority

    def should_stop_trading_based_on_market_red(self,
                                                num_pairs,
                                                majority_num,
                                                num_candles,
                                                dip_threshold):
        return self.market_is_red(num_pairs=num_pairs,
                                  majority_num=majority_num,
                                  num_candles=num_candles,
                                  dip_threshold=dip_threshold)

    def market_is_red(self, num_pairs=50, majority_num=20, num_candles=3, dip_threshold=0.015):
        pairs = self.market_info.get_active_pairs()[:num_pairs]

        num_dipped = 0

        for pair in pairs:
            try:
                old_candle = self.market_info.get_pair_candlestick(pair, ind=num_candles)
                this_candle = self.market_info.get_pair_candlestick(pair, ind=0)
            except KeyError as e:
                self.logger.error('KeyError while getting candlestick for {}:{}'.format(pair, e))
                return False

            old_price = old_candle.close
            this_price = this_candle.close

            if this_price < old_price:
                dip_amount = (old_price - this_price) / old_price
                if dip_amount >= dip_threshold:
                    num_dipped += 1

        self.logger.debug('market_red_check_num_dipped_pairs: %d', num_dipped)
        return num_dipped >= majority_num

    def should_start_trading_based_on_roi(self,
                                          backtesting_time,
                                          roi_threshold,
                                          signal_generator_class):
        trading_info = self.get_backtesting_trading_info(
            backtesting_time=backtesting_time, signal_generator_class=signal_generator_class
        )
        stats = period_statistics(trading_info=trading_info)
        return stats['roi_live'] >= roi_threshold

    def should_start_trading_based_on_recent_roi(self,
                                                 backtesting_time,
                                                 recent_roi_time,
                                                 recent_roi_threshold,
                                                 signal_generator_class):
        trading_info = self.get_backtesting_trading_info(
            backtesting_time=backtesting_time, signal_generator_class=signal_generator_class
        )

        estimated_balances_list = get_estimated_balances_list(trading_info)
        start_date = self.market_date - recent_roi_time

        start_ind = find_smaller_equal_date_index(estimated_balances_list, start_date)

        start_balance = estimated_balances_list[start_ind][1]
        end_balance = estimated_balances_list[len(estimated_balances_list) - 1][1]

        recent_roi = (end_balance - start_balance) / start_balance
        return recent_roi >= recent_roi_threshold

    def should_start_trading_based_on_market_green(self,
                                                   num_pairs,
                                                   majority_num,
                                                   num_candles,
                                                   up_threshold):
        return self.market_is_green(num_pairs=num_pairs,
                                    majority_num=majority_num,
                                    num_candles=num_candles,
                                    up_threshold=up_threshold)

    def market_is_green(self, num_pairs=50, majority_num=25, num_candles=6, up_threshold=0.00):
        pairs = self.market_info.get_active_pairs()[:num_pairs]

        num_upped = 0

        for pair in pairs:
            try:
                old_candle = self.market_info.get_pair_candlestick(pair, ind=num_candles)
                this_candle = self.market_info.get_pair_candlestick(pair, ind=0)
            except KeyError as e:
                self.logger.error('KeyError while getting candlestick for {}:{}'.format(pair, e))
                return False

            old_price = old_candle.close
            this_price = this_candle.close

            if this_price > old_price:
                up_amount = (this_price - old_price) / old_price
                if up_amount >= up_threshold:
                    num_upped += 1

        self.logger.debug('market_green_check_num_upped_pairs: %d', num_upped)
        return num_upped >= majority_num

    def get_backtesting_trading_info(self, backtesting_time, signal_generator_class):
        market_info = BacktestingMarketInfo(candlestick_store=CandlestickStore.get_instance())

        account = BacktestingAccount(market_info=market_info, balances={'BTC': 100})

        start_date = self.market_date - backtesting_time
        end_date = self.market_date

        signal_generators = [
            signal_generator_class(market_info=market_info, enable_disable=False, silent=True)
        ]
        signal_executor = SignalExecutor(market_info=market_info, account=account, silent=True)

        backtest.backtest(account=account, market_info=market_info,
                          signal_generators=signal_generators, signal_executor=signal_executor,
                          start=start_date, end=end_date, silent=True)

        return signal_executor.get_trading_info()

    def get_high_volume_pairs(self, high_volume_limit):
        return sorted(
            filter(lambda p: self.market_info.get_pair_last_24h_btc_volume(p) >= high_volume_limit,
                   self.market_info.get_active_pairs()),
            key=lambda pair: -self.market_info.get_pair_last_24h_btc_volume(pair=pair)
        )
