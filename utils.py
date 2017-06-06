from datetime import datetime

from account import Account
from currency import Currency
from marketinfo import MarketInfo
from order import Order


def backtest(account: Account, market_info: MarketInfo, strategy):
    start_date = market_info.get_min_pair_start_time()
    end_date = market_info.get_min_pair_end_time()

    print('start:', datetime.fromtimestamp(start_date))
    print('end:', datetime.fromtimestamp(end_date))

    market_info.set_market_time(start_date)
    while market_info.get_market_time() < end_date - 300 * 6:
        account.execute_orders(market_info)
        strategy.act(account, market_info)
        market_info.inc_market_time()
