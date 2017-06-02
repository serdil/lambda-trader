from currency import Currency
from currencypair import CurrencyPair
from order import Order, OrderType
from pairinfo import PairInfo


def actor_market_buyer(account, market_info):
    for pair, pair_info in market_info.pairs.items():
        actor_pair_buyer(account, pair, pair_info)

def actor_pair_buyer(account, pair: CurrencyPair, pair_info: PairInfo):
    chunk_size = account.get_estimated_balance() / 25
    price = pair_info.history[-1].close
    timestamp = pair_info.history[-1].timestamp
    if account.balances[Currency.BTC] >= chunk_size * 1.01:
        account.buy(pair.second, price, chunk_size / price)
        sell_order = Order(pair.second, OrderType.SELL, price, chunk_size / price, timestamp)
        account.new_order(sell_order)