from time import sleep

from poloniex import PoloniexError

from lambdatrader.exchanges.poloniex.polxclient import polo
from lambdatrader.utilities.utils import pair_from

DELTA = 0.00001

for orders in polo.returnOpenOrders().values():
    for order in orders:
        print('cancelling')
        polo.cancelOrder(order['orderNumber'])

balances_to_print = polo.returnBalances()
for key, value in balances_to_print.items():
    if float(value) > 0.0:
        print(key, ':', value)

while True:
    balances = polo.returnBalances()
    ticker = polo.returnTicker()
    total_amount = sum([float(value) if key != 'BTC' else 0 for key, value in balances.items()])
    print('total amount:', total_amount)
    if total_amount < DELTA:
        print('success')
        break
    for key, value in balances.items():
        if float(value) > 0.00010:
            try:
                if key != 'BTC':
                    highest_bid = float(ticker[pair_from('BTC', key)]['highestBid'])
                    lower_price = highest_bid * 0.999
                    even_lower_price = highest_bid * 0.995
                    even_lower_than_lower = highest_bid * 0.99
                    try:
                        polo.sell(pair_from('BTC', key),
                                  highest_bid, float(value),
                                  orderType='fillOrKill')
                        continue
                    except PoloniexError:
                        pass
                    print('tried to sell for', highest_bid)
                    sleep(5)
                    try:
                        polo.sell(pair_from('BTC', key),
                                  lower_price, float(value),
                                  orderType='fillOrKill')
                        print('tried to sell for', lower_price)
                        continue
                    except PoloniexError:
                        pass
                    sleep(5)
                    try:
                        polo.sell(pair_from('BTC', key),
                                  even_lower_than_lower, float(value),
                                  orderType='fillOrKill')
                        continue
                    except PoloniexError:
                        pass
                    print('tried to sell for', even_lower_than_lower)
                    print('sold')
            except Exception as e:
                print(e)
