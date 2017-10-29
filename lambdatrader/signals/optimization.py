from lambdatrader.backtesting import backtest
from lambdatrader.backtesting.account import BacktestingAccount
from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.executors.executors import SignalExecutor
from lambdatrader.history.store import CandlestickStore
from lambdatrader.signals.constants import ONE_DAY_SECONDS


class OptimizationMixin:

    last_optimized = 0

    def optimization_set_params(self, *args):
        raise NotImplementedError

    def optimization_get_params_info(self):
        raise NotImplementedError

    def optimization_get_optimization_frequency(self):
        return 7 * ONE_DAY_SECONDS

    def optimization_get_objective_function(self):
        periods = [7*ONE_DAY_SECONDS, 35*ONE_DAY_SECONDS, 91*ONE_DAY_SECONDS]
        weights = [3, 2, 1]
        max_costs = [10, 20, 30]
        # TODO implement real cost function
        cost_function = lambda: 1
        return ObjectiveFunction(signal_generator_class=self.__class__,
                                 market_date=self.get_market_date(),
                                 periods=periods,
                                 weights=weights,
                                 max_costs=max_costs,
                                 cost_function=cost_function)

    def optimization_update_parameters_if_necessary(self):
        if self.__should_optimize():
            self.optimization_set_params(*self.__optimize())
            self.last_optimized = self.get_market_date()

    def __should_optimize(self):
        return self.get_market_date() - self.last_optimized >=\
               self.optimization_get_optimization_frequency()

    def __optimize(self):
        # TODO implement optimization routine
        return [1] * self.optimization_get_params_info()['num_params']


class ObjectiveFunction:
    MAX_COST = 1000000000

    def __init__(self, signal_generator_class,
                 market_date, periods, weights, max_costs, cost_function):
        self.signal_generator_class = signal_generator_class
        self.market_date = market_date
        self.periods = periods
        self.weights = weights
        self.max_costs = max_costs
        self.cost_function = cost_function

        self.market_info = BacktestingMarketInfo(candlestick_store=CandlestickStore.get_instance())

    def __call__(self, *args, **kwargs):
        total_cost = 0
        signal_generator = self.__create_signal_generator(*args)
        for period_no, period in enumerate(self.periods):
            cost = self.__calc_period_score(signal_generator, period)
            if cost > self.max_costs[period_no]:
                cost = self.MAX_COST
            cost = cost * self.weights[period_no] / sum(self.weights)
            total_cost += cost
        return total_cost

    def __create_signal_generator(self, *args):
        signal_generator = self.signal_generator_class(self.market_info,
                                                       live=False, silent=True, optimize=False)
        signal_generator.set_parameters(*args)
        return signal_generator

    def __calc_period_score(self, signal_generator, period):
        account = BacktestingAccount(market_info=self.market_info, balances={'BTC': 100})
        signal_executor = SignalExecutor(market_info=self.market_info, account=account)
        backtest.backtest(account=account, market_info=self.market_info,
                          signal_generators=[signal_generator], signal_executor=signal_executor,
                          start=self.market_date - period, end=self.market_date)
        return self.cost_function(signal_executor.get_trading_info())
