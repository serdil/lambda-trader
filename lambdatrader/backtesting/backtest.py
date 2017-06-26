from datetime import datetime

from backtesting.marketinfo import BacktestMarketInfo

from backtesting.account import Account


def backtest(account: Account, market_info: BacktestMarketInfo, strategies, finish_before=6):
    start_date = market_info.get_min_pair_start_time()
    end_date = market_info.get_max_pair_end_time() - finish_before * 300

    print('start:', datetime.fromtimestamp(start_date))
    print('end:', datetime.fromtimestamp(end_date))

    market_info.set_market_time(start_date)
    while market_info.get_market_time() < end_date:
        account.execute_orders(market_info)
        for strategy in strategies:
            strategy.act(account, market_info)
        market_info.inc_market_time()
