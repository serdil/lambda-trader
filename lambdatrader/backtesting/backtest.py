from datetime import datetime

from backtesting.marketinfo import BacktestMarketInfo

from backtesting.account import Account
from utils import date_ceil, date_floor


def backtest(account: Account, market_info: BacktestMarketInfo, strategy, start=0, end=9999999999):
    normalized_start = date_ceil(start)
    normalized_end = date_floor(end)
    start_date = max(market_info.get_min_pair_start_time(), normalized_start)
    end_date = min(market_info.get_min_pair_end_time(), normalized_end)

    print('start:', datetime.fromtimestamp(start_date))
    print('end:', datetime.fromtimestamp(end_date))

    strategy.set_history_start(start_date)
    strategy.set_history_end(end_date)

    market_info.set_market_time(start_date)
    while market_info.get_market_time() < end_date:
        account.execute_orders(market_info)
        strategy.act(account, market_info)
        market_info.inc_market_time()
