from polx.poloniexclient import polo
from utils import pair_from

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
                    polo.sell(pair_from('BTC', key), float(ticker[pair_from('BTC', key)]['highestBid']), float(value), orderType='fillOrKill')
                    print('sold')
            except Exception as e:
                print(e)
