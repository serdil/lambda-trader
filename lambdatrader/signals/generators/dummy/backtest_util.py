from pprint import pprint

from lambdatrader.backtesting import backtest
from lambdatrader.backtesting.account import BacktestingAccount
from lambdatrader.evaluation.utils import period_statistics, statistics_over_periods
from lambdatrader.executors.executors import SignalExecutor


def print_descriptors(signal_generators):
    print()
    print('Descriptors:')
    for signal_generator in signal_generators:
        try:
            pprint(signal_generator.get_algo_descriptor())
        except AttributeError:
            print('Signal generator has no descriptor.')


def do_backtest(signal_generator, market_info, start_date, end_date):
    account = BacktestingAccount(market_info=market_info, balances={'BTC': 100})

    signal_generators = [signal_generator]

    signal_executor = SignalExecutor(market_info=market_info, account=account)

    print_descriptors(signal_generators)

    backtest.backtest(account=account, market_info=market_info, signal_generators=signal_generators,
                      signal_executor=signal_executor, start=start_date, end=end_date)

    print('Signal Generators Used:')

    for signal_generator in signal_generators:
        pprint(signal_generator.__class__.__dict__)
        pprint(signal_generator.__dict__)

    print()
    print('Backtest Complete!')

    print()
    print('Estimated Balance:', account.get_estimated_balance())

    print_descriptors(signal_generators)

    trading_info = signal_executor.get_trading_info()

    stats = period_statistics(trading_info=trading_info)

    print()
    print('Statistics over whole trading period:')
    pprint(stats)

    stats_over_weekly_periods = statistics_over_periods(trading_info=trading_info, period_days=7)

    print()
    print('Statistics over weekly periods:')
    pprint(stats_over_weekly_periods)

    stats_over_monthly_periods = statistics_over_periods(trading_info=trading_info, period_days=30)

    print()
    print('Statistics over monthly periods:')
    pprint(stats_over_monthly_periods)
