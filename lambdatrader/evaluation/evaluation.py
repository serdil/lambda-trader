from _bisect import bisect_left, bisect_right
from operator import attrgetter

import numpy
from blist import sorteddict


class Evaluator:

    def __init__(self, trading_info):
        self.__trading_info = trading_info
        self.__trades = sorted(trading_info.trades, key=attrgetter('start_date'))
        self.__trade_start_dates = [trade.start_date for trade in self.__trades]
        self.__balances = sorteddict(trading_info.balances)

    def get_trading_info(self):
        return self.__trading_info

    def calc_stats_for_period(self, start_date, end_date):
        num_of_trades = self.__num_of_trades(start_date, end_date)
        maximum_drawdown = self.__max_drawdown(start_date, end_date)
        roi = self.__roi(start_date, end_date)
        success_rate = self.__success_rate(start_date, end_date)
        longest_drawdown_period = self.__longest_drawdown_period(start_date, end_date)

        return {
            'num_of_trades': num_of_trades,
            'maximum_drawdown': maximum_drawdown,
            'roi': roi,
            'success_rate': success_rate,
            'longest_drawdown_period': longest_drawdown_period
        }

    def __num_of_trades(self, start_date, end_date):
        return sum([1 for _ in self.__period_trades(start_date, end_date)])

    def __max_drawdown(self, start_date, end_date):
        max_so_far = 0
        max_drawdown = 0

        for balance in self.__period_balances(start_date, end_date):
            if balance > max_so_far:
                max_so_far = balance
            else:
                current_drawdown = (max_so_far - balance) / max_so_far
                max_drawdown = max(max_drawdown, current_drawdown)

        return max_drawdown

    def __roi(self, start_date, end_date):
        start_balance = self.__balance_after(start_date)
        end_balance = self.__balance_before(end_date)

        return (end_balance - start_balance) / start_balance

    def __success_rate(self, start_date, end_date):
        num_positive = 0
        num_non_positive = 0
        for trade in self.__period_trades(start_date, end_date):
            if trade.profit_amount > 0:
                num_positive += 1
            else:
                num_non_positive += 1

        return num_positive / num_positive + num_non_positive

    def __longest_drawdown_period(self, start_date, end_date):
        dates_balances = [(date, balance) for date, balance
         in self.__period_dates_and_balances(start_date, end_date)]

        longest_drawdown = 0

        for i, (date1, balance1) in enumerate(dates_balances):
            for j, (date2, balance2) in enumerate(reversed(dates_balances), start=1):
                if len(dates_balances) - j == i:
                    break
                if balance2 < balance1:
                    longest_drawdown = max(longest_drawdown, date2 - date1)
                    break

        return longest_drawdown

    def __period_trades(self, start_date, end_date):
        start_ind = bisect_left(self.__trade_start_dates, start_date)
        end_ind = bisect_right(self.__trade_start_dates, end_date)

        for i in range(start_ind, end_ind):
            yield self.__trades[i]

    def __period_balances(self, start_date, end_date):
        for date, balance in self.__period_dates_and_balances(start_date, end_date):
            yield balance

    def __period_dates_and_balances(self, start_date, end_date):
        keys = self.__balances.keys()
        start_ind = keys.bisect_left(start_date)
        end_ind = keys.bisect_right(end_date)

        for i in range(start_ind, end_ind):
            yield keys[i], self.__balances[keys[i]]

    def __balance_after(self, date):
        keys = self.__balances.keys()
        return self.__balances[keys[keys.bisect_left(date)]]

    def __balance_before(self, date):
        keys = self.__balances.keys()
        key_ind = keys.bisect_right(date)
        if key_ind == len(keys):
            key_ind -= 1
        return self.__balances[keys[key_ind]]

    @classmethod
    def calc_stats_over_periods(cls, period_stats):
        longest_drawdown_period = cls.__periods_longest_drawdown_period(period_stats)
        median_roi = cls.__periods_median_roi(period_stats)
        first_q_roi = cls.__periods_first_first_q_roi(period_stats)
        minimum_roi = cls.__periods_minimum_roi(period_stats)
        median_num_of_trades = cls.__periods_median_num_of_trades(period_stats)
        first_q_num_of_trades = cls.__periods_first_q_num_of_trades(period_stats)
        minimum_num_of_trades = cls.__periods_minimum_num_of_trades(period_stats)
        median_success_rate = cls.__periods_median_success_rate(period_stats)
        first_q_success_rate = cls.__periods_first_q_success_rate(period_stats)
        minimum_success_rate = cls.__periods_minimum_success_rate(period_stats)

        return {
            'longest_drawdown_period': longest_drawdown_period,
            'median_roi': median_roi,
            'first_q_roi': first_q_roi,
            'minimum_roi': minimum_roi,
            'median_num_of_trades': median_num_of_trades,
            'first_q_num_of_trades': first_q_num_of_trades,
            'minimum_num_of_trades': minimum_num_of_trades,
            'median_success_rate': median_success_rate,
            'first_q_success_rate': first_q_success_rate,
            'minimum_success_rate': minimum_success_rate
        }

    @classmethod
    def __periods_longest_drawdown_period(cls, period_stats):
        longest_subseq_len = 0
        subseq_len = 0

        rois = cls.__periods_roi_list(period_stats)

        for i, roi in enumerate(rois):
            if roi < 0:
                subseq_len += 1
                if subseq_len > longest_subseq_len:
                    longest_subseq_len = subseq_len
            else:
                subseq_len = 0

        return longest_subseq_len

    @classmethod
    def __periods_median_roi(cls, period_stats):
        rois = cls.__periods_roi_list(period_stats)
        return numpy.median(rois)

    @classmethod
    def __periods_first_first_q_roi(cls, period_stats):
        rois = cls.__periods_roi_list(period_stats)
        return numpy.percentile(rois, 25)

    @classmethod
    def __periods_minimum_roi(cls, period_stats):
        rois = cls.__periods_roi_list(period_stats)
        return min(rois)

    @classmethod
    def __periods_median_num_of_trades(cls, period_stats):
        num_of_trades = cls.__periods_num_of_trades_list(period_stats)
        return numpy.median(num_of_trades)

    @classmethod
    def __periods_first_q_num_of_trades(cls, period_stats):
        num_of_trades = cls.__periods_num_of_trades_list(period_stats)
        return numpy.percentile(num_of_trades, 25)

    @classmethod
    def __periods_minimum_num_of_trades(cls, period_stats):
        num_of_trades = cls.__periods_num_of_trades_list(period_stats)
        return min(num_of_trades)

    @classmethod
    def __periods_median_success_rate(cls, period_stats):
        success_rates = cls.__periods_success_rate_list(period_stats)
        return numpy.median(success_rates)

    @classmethod
    def __periods_first_q_success_rate(cls, period_stats):
        success_rates = cls.__periods_success_rate_list(period_stats)
        return numpy.percentile(success_rates, 25)

    @classmethod
    def __periods_minimum_success_rate(cls, period_stats):
        success_rates = cls.__periods_success_rate_list(period_stats)
        return min(success_rates)
    
    @staticmethod
    def __periods_roi_list(period_stats):
        return [stats['roi'] for stats in period_stats]
    
    @staticmethod
    def __periods_num_of_trades_list(period_stats):
        return [stats['num_of_trades'] for stats in period_stats]

    @staticmethod
    def __periods_success_rate_list(period_stats):
        return [stats['success_rate'] for stats in period_stats]
