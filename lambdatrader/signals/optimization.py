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
        # TODO write objective function
        return lambda: 1

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
