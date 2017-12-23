import itertools
from datetime import datetime

from lambdatrader.backtesting.account import BacktestingAccount
from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.utilities.utils import date_ceil, date_floor


def backtest(account: BacktestingAccount, market_info: BacktestingMarketInfo,
             signal_generators, signal_executor, start=0, end=9999999999):
    normalized_start = date_ceil(start)
    normalized_end = date_floor(end)
    start_date = max(market_info.get_min_pair_start_time(), normalized_start)
    end_date = min(market_info.get_max_pair_end_time(), normalized_end)

    print('start:', datetime.fromtimestamp(start_date))
    print('end:', datetime.fromtimestamp(end_date))

    market_info.set_market_date(start_date)

    signal_executor.set_history_end(end_date)
    signal_executor.set_history_start(start_date)

    tracked_signals = []
    while market_info.get_market_date() < end_date:
        account.execute_orders()
        signals = \
            list(itertools.chain.from_iterable([list(generator.generate_signals(tracked_signals=tracked_signals))
                                                for generator in signal_generators]))
        tracked_signals = signal_executor.act(signals=signals)
        market_info.inc_market_time()

    account.execute_orders()
