import math

import numpy as np
import rbfopt

from lambdatrader.evaluation.utils import period_stats_roi_max_drawdown_score
from lambdatrader.backtesting import backtest
from lambdatrader.backtesting.account import BacktestingAccount
from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.executors.executors import SignalExecutor
from lambdatrader.history.store import CandlestickStore
from lambdatrader.signals.constants import ONE_DAY_SECONDS


class ObjectiveFunction:
    MAX_COST = 1000000000

    def __init__(self, signal_generator_class,
                 market_date, periods, weights, max_costs, cost_functions):
        self.signal_generator_class = signal_generator_class
        self.market_date = market_date
        self.periods = periods
        self.weights = weights
        self.max_costs = max_costs
        self.cost_functions = cost_functions

        self.market_info = BacktestingMarketInfo(candlestick_store=CandlestickStore.get_instance())

    def __call__(self, *args, **kwargs):
        print('params:', args[0])
        total_cost = 0
        signal_generator = self.__create_signal_generator(args[0])
        for period_no, period in enumerate(self.periods):
            cost_function = self.cost_functions[period_no]
            cost = self.__calc_period_score(signal_generator, period, cost_function)
            if cost > self.max_costs[period_no]:
                cost = self.MAX_COST
            cost = cost * self.weights[period_no] / sum(self.weights)
            total_cost += cost
        return total_cost

    def __create_signal_generator(self, params):
        signal_generator = self.signal_generator_class(self.market_info,
                                                       live=False, silent=True, optimize=False)
        signal_generator.set_params(*params)
        return signal_generator

    def __calc_period_score(self, signal_generator, period, cost_function):
        start_date = self.market_date-period
        self.market_info.set_market_date(start_date)
        account = BacktestingAccount(market_info=self.market_info, balances={'BTC': 100})
        signal_executor = SignalExecutor(market_info=self.market_info, account=account, silent=True)
        backtest.backtest(account=account, market_info=self.market_info,
                          signal_generators=[signal_generator], signal_executor=signal_executor,
                          start=self.market_date-period, end=self.market_date, silent=True)
        return cost_function(signal_executor.get_trading_info())


class OptimizationMixin:

    MAX_EVALUATIONS = 1000

    last_optimized = 0

    def optimization_set_params(self, *args):
        raise NotImplementedError

    def optimization_get_params_info(self):
        raise NotImplementedError

    def optimization_get_optimization_frequency(self):
        return 7 * ONE_DAY_SECONDS

    def optimization_get_optimization_periods_info(self):
        return {
            'periods': [7*ONE_DAY_SECONDS, 35*ONE_DAY_SECONDS, 91*ONE_DAY_SECONDS],
            'weights': [3, 2, 1],
            'max_costs': [10, 10, 10],
            'cost_functions': [self.optimization_get_cost_function()] * 3
        }

    def optimization_get_cost_function(self):
        return lambda trading_info: \
            math.e**(-period_stats_roi_max_drawdown_score(trading_info))

    def optimization_get_objective_function(self):
        periods_info = self.optimization_get_optimization_periods_info()
        num_periods = len(periods_info['periods'])
        cost_functions = periods_info['cost_functions'] \
            if 'cost_functions' in periods_info\
            else [self.optimization_get_cost_function()] * num_periods
        return ObjectiveFunction(signal_generator_class=self.__class__,
                                 market_date=self.get_market_date(),
                                 periods=periods_info['periods'],
                                 weights=periods_info['weights'],
                                 max_costs=periods_info['max_costs'],
                                 cost_functions=cost_functions)

    def optimization_update_parameters_if_necessary(self):
        if self.__should_optimize():
            self.optimization_set_params(*self.__optimize())
            self.last_optimized = self.get_market_date()

    def __should_optimize(self):
        return self.get_market_date() - self.last_optimized >=\
               self.optimization_get_optimization_frequency()

    def __optimize(self):
        objective_function = self.optimization_get_objective_function()
        num_params = self.__get_num_params()
        min = self.__get_params_min_np()
        max = self.__get_params_max_np()
        types = self.__get_params_types_np()

        bb = rbfopt.RbfoptUserBlackBox(num_params, min, max,
                                       types, objective_function)
        settings = rbfopt.RbfoptSettings(max_evaluations=self.MAX_EVALUATIONS)
        alg = rbfopt.RbfoptAlgorithm(settings, bb)

        val, params, itercount, evalcount, fast_evalcount = alg.optimize()
        params = self.__convert_params_to_original_type(params, self.__get_params_types())
        return params

    def __get_num_params(self):
        return self.optimization_get_params_info()['num_params']

    def __get_params_min_np(self):
        return np.array(self.optimization_get_params_info()['min'])

    def __get_params_max_np(self):
        return np.array(self.optimization_get_params_info()['max'])

    def __get_params_types_np(self):
        return np.array(self.__get_params_types())

    def __get_params_types(self):
        return self.optimization_get_params_info()['type']

    @staticmethod
    def __convert_params_to_original_type(params, types):
        orig_params = []
        for i, param in enumerate(params):
            if types[i] == 'I':
                orig_params.append(int(param))
            elif types[i] == 'F':
                orig_params.append(param)
        return orig_params
