from currency import Currency
from currencypair import CurrencyPair
from marketinfo import MarketInfo
from order import Order, OrderType


def actor_market_buyer(account, market_info):
    for pair, _ in market_info.pairs():
        actor_pair_buyer(account, market_info, pair)


def actor_pair_buyer(account, market_info: MarketInfo, pair: CurrencyPair):
    chunk_size = account.get_estimated_balance(market_info) / 25
    if chunk_size >= 0.00011:
        latest_candlestick = market_info.get_pair_latest_candlestick(pair)
        price = latest_candlestick.close
        timestamp = latest_candlestick.timestamp
        if account.get_balance(Currency.BTC) >= chunk_size * 1.0001:
            account.buy(pair.second, price, chunk_size / price)
            sell_order = Order(pair.second, OrderType.SELL, price * 1.02, account.get_balance(pair.second), timestamp)
            account.new_order(sell_order)
