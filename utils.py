from typing import Callable, Iterable, List

from account import Account
from marketinfo import MarketInfo
from order import Order
from pairinfo import PairInfo


def find_min_timestamp(market_info):
    min_timestamp = int(2**32)
    for _, pair_info in market_info.pairs.items():
        if pair_info.history:
            if pair_info.history[0] < min_timestamp:
                min_timestamp = pair_info.history[0]
    return min_timestamp


def find_max_timestamp(market_info):
    max_timestamp = -1
    for _, pair_info in market_info.pairs.items():
        if pair_info.history:
            if pair_info.history[0] > max_timestamp:
                max_timestamp = pair_info.history[0]
    return max_timestamp


def get_pair_info_slice_upto(pair_info, timestamp):
    snapshot = PairInfo(pair_info.currency_pair, [])
    history = pair_info.history
    for candlestick in history:
        if candlestick.timestamp <= timestamp:
            snapshot.history.append(candlestick)
    return snapshot


def get_market_slice_upto(market_info, timestamp):
    snapshot = MarketInfo()
    for pair, pair_info in market_info.pairs.items():
        snapshot.pairs[pair] = get_pair_info_slice_upto(pair_info, timestamp)


def get_market_snapshots(market_info: MarketInfo) -> Iterable(MarketInfo):
    min_timestamp = find_min_timestamp(market_info)
    max_timestamp = find_max_timestamp(market_info)
    timestamp = min_timestamp
    while(timestamp <= max_timestamp):
        yield get_market_slice_upto(market_info, timestamp)
        timestamp += 300


def backtest(account: Account, market_info: MarketInfo, act: Callable([Account, MarketInfo], Iterable(Order))):
    for market_snapshot in get_market_snapshots(market_info):
        account.execute_orders(market_snapshot)
        act(account, market_snapshot)
