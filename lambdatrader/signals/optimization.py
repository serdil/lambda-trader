import operator
from functools import reduce

import numpy as np
from platypus import NSGAII, Problem, Real, Integer

from lambdatrader.backtesting import backtest
from lambdatrader.backtesting.account import BacktestingAccount
from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.candlestick_stores.cachingstore import ChunkCachingCandlestickStore
from lambdatrader.evaluation.utils import (
    get_costs,
)
from lambdatrader.executors.executors import SignalExecutor
from lambdatrader.signals.constants import ONE_DAY_SECONDS


class ObjectiveFunction:
    MAX_COST = 1000000000

    def __init__(self, signal_generator_class,
                 market_date, periods, weights, max_costs, costs_functions):
        self.signal_generator_class = signal_generator_class
        self.market_date = market_date
        self.periods = periods
        self.weights = weights
        self.max_costs = max_costs
        self.costs_functions = costs_functions

        self.market_info = BacktestingMarketInfo(candlestick_store=ChunkCachingCandlestickStore.get_instance())

    def __call__(self, *args, **kwargs):
        solution = args[0]
        variables = solution.variables
        print('variables:', variables)
        objectives = solution.objectives
        for i in range(len(objectives)):
            objectives[i] = 0
        signal_generator = self.__create_signal_generator(variables)
        for period_no, period in enumerate(self.periods):
            costs_function = self.costs_functions[period_no]
            costs = self.__calc_period_costs(signal_generator, period, costs_function)
            for i, cost in enumerate(costs):
                cost = cost * self.weights[period_no] / sum(self.weights)
                objectives[i] += cost

    def __create_signal_generator(self, params):
        signal_generator = self.signal_generator_class(self.market_info,
                                                       live=False, silent=True, optimize=False)
        signal_generator.set_params(*params)
        return signal_generator

    def __calc_period_costs(self, signal_generator, period, costs_function):
        start_date = self.market_date-period
        self.market_info.set_market_date(start_date)
        account = BacktestingAccount(market_info=self.market_info, balances={'BTC': 100})
        signal_executor = SignalExecutor(market_info=self.market_info, account=account, silent=True)
        backtest.backtest(account=account, market_info=self.market_info,
                          signal_generators=[signal_generator], signal_executor=signal_executor,
                          start=self.market_date-period, end=self.market_date, silent=True)
        return costs_function(signal_executor.get_trading_info())


class TradingProblem(Problem):

    def __init__(self, nvars, nobjs, types, objective_function):
        super().__init__(nvars, nobjs)
        self.types[:] = types
        self.objective_function = objective_function

    def evaluate(self, solution):
        self.objective_function(solution)


class OptimizationMixin:

    MAX_EVALUATIONS = 10000

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
            'costs_functions': [self.optimization_get_costs_function()] * 3,
        }

    def optimization_get_costs_function_num_costs(self):
        return 2

    def optimization_get_costs_function(self):
        return self._costs_function

    def _costs_function(self, trading_info):
        costs = get_costs(trading_info)
        return [costs['roi_live_cost'], costs['max_drawdown_live_cost']]

    def optimization_get_objective_function(self):
        periods_info = self.optimization_get_optimization_periods_info()
        num_periods = len(periods_info['periods'])
        costs_functions = periods_info['cost_functions'] \
            if 'costs_functions' in periods_info \
            else [self.optimization_get_costs_function()] * num_periods
        return ObjectiveFunction(signal_generator_class=self.__class__,
                                 market_date=self.get_market_date(),
                                 periods=periods_info['periods'],
                                 weights=periods_info['weights'],
                                 max_costs=periods_info['max_costs'],
                                 costs_functions=costs_functions)

    def optimization_update_parameters_if_necessary(self):
        if self.__should_optimize():
            self.optimization_set_params(*self.__optimize())
            self.last_optimized = self.get_market_date()

    def optimization_select_best_solution(self, solutions):
        return min(solutions, key=lambda sol: reduce(operator.mul, sol.objectives, 1))

    def __should_optimize(self):
        return self.get_market_date() - self.last_optimized >= \
               self.optimization_get_optimization_frequency()

    def __optimize(self):
        objective_function = self.optimization_get_objective_function()
        num_params = self.__get_num_params()
        num_costs = self.__get_num_costs()
        min = self.__get_params_min()
        max = self.__get_params_max()
        types = self.__get_params_types()

        platypus_types = self.__convert_min_max_types_to_platypus_types(min, max, types)

        algorithm = NSGAII(
            TradingProblem(num_params, num_costs, platypus_types, objective_function))
        algorithm.run(self.MAX_EVALUATIONS)
        best_solution = self.optimization_select_best_solution(algorithm.result)
        print(best_solution.variables, best_solution.objectives) # TODO remove
        return best_solution.variables

    def __get_num_params(self):
        return self.optimization_get_params_info()['num_params']

    def __get_num_costs(self):
        return self.optimization_get_costs_function_num_costs()

    def __get_params_min_np(self):
        return np.array(self.__get_params_min())

    def __get_params_min(self):
        return self.optimization_get_params_info()['min']

    def __get_params_max_np(self):
        return np.array(self.__get_params_max())

    def __get_params_max(self):
        return self.optimization_get_params_info()['max']

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

    @staticmethod
    def __convert_min_max_types_to_platypus_types(min, max, types):
        platypus_types = []
        for i, type in enumerate(types):
            if type == 'I':
                platypus_types.append(Integer(min[i], max[i]))
            elif type == 'F':
                platypus_types.append(Real(min[i], max[i]))
        return platypus_types
