from typing import Callable, Iterable
from datetime import datetime

from account import Account
from currency import Currency
from marketinfo import MarketInfo
from order import Order


def backtest(account: Account, market_info: MarketInfo, act: Callable[[Account, MarketInfo], Iterable[Order]]):
    start_date = market_info.get_min_pair_start_time()
    end_date = market_info.get_max_pair_end_time()
    print('start:', datetime.fromtimestamp(start_date))
    print('end:', datetime.fromtimestamp(end_date))
    market_info.set_market_time(start_date)
    while market_info.get_market_time() < end_date:
        account.execute_orders(market_info)
        act(account, market_info)
        market_info.inc_market_time()
        #print('balance: ', account.get_estimated_balance(market_info))
        #print('btc: ', account.get_balance(Currency.BTC))
        #print('orders: ', len(list(account.get_orders())))
        #print(list(account.get_orders()))
