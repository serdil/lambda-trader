from typing import Callable, Iterable

from account import Account
from currency import Currency
from marketinfo import MarketInfo
from order import Order


def backtest(account: Account, market_info: MarketInfo, act: Callable[[Account, MarketInfo], Iterable[Order]]):
    min_timestamp = market_info.get_min_pair_start_time()
    max_timestamp = market_info.get_max_pair_end_time()
    market_info.set_market_time(min_timestamp)
    while market_info.get_market_time() < max_timestamp:
        account.execute_orders(market_info)
        act(account, market_info)
        market_info.inc_market_time()
        #print('balance: ', account.get_estimated_balance(market_info))
        #print('btc: ', account.get_balance(Currency.BTC))
        #print('orders: ', len(list(account.get_orders())))
        #print(list(account.get_orders()))
