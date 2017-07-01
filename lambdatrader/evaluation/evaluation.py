from _bisect import bisect_left, bisect_right

from blist import sorteddict

class Evaluator:

    def __init__(self, trading_info):
        self.__trading_info = trading_info
        self.__trades = sorted(trading_info.trades)
        self.__trade_start_dates = [trade.start_date for trade in self.__trades]
        self.__balances = sorteddict(trading_info.balances)

    def get_trading_info(self):
        return self.__trading_info

    def calc_metrics_for_period(self, start_date, end_date):
        number_of_trades = self.__number_of_trades(start_date, end_date)
        maximum_drawdown = self.__max_drawdown(start_date, end_date)
        roi = self.__roi(start_date, end_date)
        success_rate = self.__success_rate(start_date, end_date)
        shortest_no_drawdown_window = self.__shortest_no_drawdown_window(start_date, end_date)

        return {
            'number_of_trades': number_of_trades,
            'maximum_drawdown': maximum_drawdown,
            'roi': roi,
            'success_rate': success_rate,
            'shortest_no_drawdown_window': shortest_no_drawdown_window
        }

    def __number_of_trades(self, start_date, end_date):
        return sum([1 for _ in self.__period_trades(start_date, end_date)])

    def __max_drawdown(self, start_date, end_date):
        max_so_far = 0
        min_since_max_so_far = max_so_far
        max_drawdown = 0

        for balance in self.__period_balances(start_date, end_date):
            max_so_far = max(max_so_far, balance)
            min_since_max_so_far = min(min_since_max_so_far, balance)
            max_drawdown = min(max_drawdown, min_since_max_so_far)

        return max_drawdown

    def __roi(self, start_date, end_date):
        pass

    def __success_rate(self, start_date, end_date):
        pass

    def __shortest_no_drawdown_window(self, start_date, end_date):
        pass

    def __period_trades(self, start_date, end_date):
        start_ind = bisect_left(self.__trade_start_dates, start_date)
        end_ind = bisect_right(self.__trade_start_dates, end_date)

        for i in range(start_ind, end_ind):
            yield self.__trades[i]

    def __period_balances(self, start_date, end_date):
        keys = self.__balances.keys()
        start_ind = keys.bisect_left(start_date)
        end_ind = keys.bisect_right(end_date)

        for i in range(start_ind, end_ind):
            yield self.__balances[keys[i]]

    @classmethod
    def calc_metrics_over_periods(cls, period_metrics):
        shortest_no_drawdown_window = cls.__periods_shortest_no_drawdown_window(period_metrics)
        median_roi = cls.__periods_median_roi(period_metrics)
        first_q_roi = cls.__periods_first_first_q_roi(period_metrics)
        minimum_roi = cls.__periods_minimum_roi(period_metrics)
        median_num_of_trades = cls.__periods_median_num_of_trades(period_metrics)
        first_q_num_of_trades = cls.__periods_first_q_num_of_trades(period_metrics)
        minimum_num_of_trades = cls.__periods_minimum_num_of_trades(period_metrics)
        median_success_rate = cls.__periods_median_success_rate(period_metrics)
        first_q_success_rate = cls.__periods_first_q_success_rate(period_metrics)
        minimum_success_rate = cls.__periods_minimum_success_rate(period_metrics)

        return {
            'shortest_no_drawdown_window': shortest_no_drawdown_window,
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
    def __periods_shortest_no_drawdown_window(cls, period_metrics):
        pass

    @classmethod
    def __periods_median_roi(cls, period_metrics):
        pass

    @classmethod
    def __periods_first_first_q_roi(cls, period_metrics):
        pass

    @classmethod
    def __periods_minimum_roi(cls, period_metrics):
        pass

    @classmethod
    def __periods_median_num_of_trades(cls, period_metrics):
        pass

    @classmethod
    def __periods_first_q_num_of_trades(cls, period_metrics):
        pass

    @classmethod
    def __periods_minimum_num_of_trades(cls, period_metrics):
        pass

    @classmethod
    def __periods_median_success_rate(cls, period_metrics):
        pass

    @classmethod
    def __periods_first_q_success_rate(cls, period_metrics):
        pass

    @classmethod
    def __periods_minimum_success_rate(cls, period_metrics):
        pass
