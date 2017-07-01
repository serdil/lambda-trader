from blist import sorteddict

class Evaluator:

    def __init__(self, trading_info):
        self.__trading_info = trading_info
        self.__trades = sorted(trading_info.trades)
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
        pass

    def __max_drawdown(self, start_date, end_date):
        pass

    def __roi(self, start_date, end_date):
        pass

    def __success_rate(self, start_date, end_date):
        pass

    def __shortest_no_drawdown_window(self, start_date, end_date):
        pass

    def __iterate_over_period_trades(self, start_date, end_date):
        pass

    def __iterate_over_period_balances(self, start_date, end_date):
        pass

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
