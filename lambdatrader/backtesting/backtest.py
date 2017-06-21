from datetime import datetime

from backtesting.marketinfo import BacktestMarketInfo

from backtesting.account import Account


def backtest(account: Account, market_info: BacktestMarketInfo, strategy):
    start_date = market_info.get_min_pair_start_time()
    end_date = market_info.get_min_pair_end_time()

    print('start:', datetime.fromtimestamp(start_date))
    print('end:', datetime.fromtimestamp(end_date))

    market_info.set_market_time(start_date)
    while market_info.get_market_time() < end_date:
        account.execute_orders(market_info)
        strategy.act(account, market_info)
        market_info.inc_market_time()
