from datetime import datetime

from currency import Currency
from currencypair import CurrencyPair
from marketinfo import MarketInfo
from order import Order, OrderType


def cancel_old_orders(account, market_info):
    order_ids_to_cancel = []
    for order in account.get_open_orders():
        if market_info.get_market_time() - order.timestamp >= 3 * 24 * 3600:
            order_ids_to_cancel.append(order.id)
    for id in order_ids_to_cancel:
        print('cancelling')
        order = account.get_order(id)
        account.cancel_order(id)
        price = market_info.get_pair_latest_candlestick(CurrencyPair(Currency.BTC, order.currency)).close
        account.sell(order.currency, price, order.amount)

def get_high_volume_pairs(market_info):
    return filter(lambda kv: market_info.get_pair_last_24h_btc_volume(kv[0]) >= 1000, market_info.pairs())

def actor_market_buyer(account, market_info):
    cancel_old_orders(account, market_info)

    high_volume_pairs = sorted(list(get_high_volume_pairs(market_info)),
                               key=lambda kv: -market_info.get_pair_last_24h_btc_volume(kv[0]))
    if len(high_volume_pairs) >= 10:
        for pair, _ in high_volume_pairs:
            actor_pair_buyer(account, market_info, pair)



def actor_pair_buyer(account, market_info: MarketInfo, pair: CurrencyPair):
    chunk_size = account.get_estimated_balance(market_info) / 25
    #print(datetime.fromtimestamp(market_info.get_market_time()), pair.second, market_info.get_pair_last_24h_btc_volume(pair))
    if chunk_size >= 0.00011:
        latest_candlestick = market_info.get_pair_latest_candlestick(pair)
        price = latest_candlestick.close
        timestamp = latest_candlestick.timestamp
        if account.get_balance(Currency.BTC) >= chunk_size * 1.0001:
            account.buy(pair.second, price, chunk_size / price)
            sell_order = Order(pair.second, OrderType.SELL, price * 1.02, account.get_balance(pair.second), timestamp)
            print(datetime.fromtimestamp(market_info.get_market_time()), sell_order)
            print('balance', account.get_estimated_balance(market_info))
            print('open orders:', len(list(account.get_open_orders())))
            account.new_order(sell_order)
