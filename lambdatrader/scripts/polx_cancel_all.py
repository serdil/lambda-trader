from time import sleep

from lambdatrader.polx.polxclient import polo

from lambdatrader.utils import pair_from

DELTA = 0.00001

for orders in polo.returnOpenOrders().values():
    for order in orders:
        print('cancelling')
        polo.cancelOrder(order['orderNumber'])

while True:
    balances = polo.returnBalances()
    ticker = polo.returnTicker()
    total_amount = sum([float(value) if key != 'BTC' else 0 for key, value in balances.items()])
    print(total_amount)
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
                    polo.sell(pair_from('BTC', key),
                              highest_bid, float(value),
                              orderType='fillOrKill')
                    print('tried to sell for', highest_bid)
                    sleep(5)
                    polo.sell(pair_from('BTC', key),
                              lower_price, float(value),
                              orderType='fillOrKill')
                    print('tried to sell for', lower_price)
                    sleep(5)
                    polo.sell(pair_from('BTC', key),
                              even_lower_than_lower, float(value),
                              orderType='fillOrKill')
                    print('tried to sell for', even_lower_than_lower)
                    print('sold')
            except Exception as e:
                print(e)
