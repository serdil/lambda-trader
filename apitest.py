from poloniexclient import polo


### PUBLIC METHODS ###

# README examples

print(polo('returnTicker'))

#while True:
#    print(polo('returnTicker')['BTC_ETH'])

print(polo.marketTradeHist('BTC_ETH'))

# API Docs

print(polo.return24hVolume())

print(polo.returnOrderBook('BTC_ETH', 20))

print(polo.returnCurrencies())

# Private Methods

print(polo.returnBalances())

print(polo.returnCompleteBalances())

print(polo.returnDepositAddresses())

print(polo.returnDepositsWithdrawals())

print(polo.returnOpenOrders())

print(polo.returnTradeHistory())

# CAUTION

print(polo.returnBalances())

if float(polo.returnBalances()['BTC']) >= 0.00011:
    buy_result = polo.buy('BTC_ETH', 0.01, 0.011)
    print(buy_result)


open_orders = polo.returnOpenOrders()

for orders in open_orders.values():
    for order in orders:
        if order['type'] == 'buy':
            print('cancelling', order)
            polo.cancelOrder(order['orderNumber'])

