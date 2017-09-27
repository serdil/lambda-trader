from poloniex import PoloniexError

from lambdatrader.polx.polxclient import polo

### PUBLIC METHODS ###

# README examples

print(polo('returnTicker'))

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


enter_next_block = False

if enter_next_block:
    if float(polo.returnBalances()['BTC']) >= 0.00011:
        try:
            buy_result = polo.buy('BTC_ETH', 0.01, 0.011, orderType='fillOrKill')
            print(buy_result)
        except PoloniexError as e:
            if str(e) == 'Unable to fill order completely.':
                print(e)
            else:
                raise e


enter_next_block = False

if enter_next_block:
    open_orders = polo.returnOpenOrders()

    for orders in open_orders.values():
        for order in orders:
            if order['type'] == 'buy':
                print('cancelling', order)
                polo.cancelOrder(order['orderNumber'])

