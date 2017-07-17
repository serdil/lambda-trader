from datetime import datetime
from typing import Iterable

from lambdatrader.config import (
    EXECUTOR__NUM_CHUNKS,
    EXECUTOR__MIN_CHUNK_SIZE,
)
from models.order import Order, OrderType
from models.trade import Trade
from models.tradesignal import TradeSignal
from models.tradinginfo import TradingInfo
from utils import pair_from, pair_second


class BaseSignalExecutor:
    def __init__(self):
        self.__trade_starts = {}
        self.__trades = []
        self.__history_start = None
        self.__history_end = None
        self.__balances = {}

    def set_history_start(self, date):
        self.__history_start = date

    def set_history_end(self, date):
        self.__history_end = date

    def declare_trade_start(self, date, trade_number):
        self.__trade_starts[trade_number] = date

    def declare_trade_end(self, date, trade_id, profit_amount):
        start_date = self.__trade_starts[trade_id]
        end_date = date
        trade = Trade(_id=trade_id, start_date=start_date, end_date=end_date, profit=profit_amount)
        self.__trades.append(trade)

    def declare_balance(self, date, balance):
        self.__balances[date] = balance

    def get_trading_info(self):
        return TradingInfo(history_start=self.__history_start, history_end=self.__history_end,
                           balances=dict(self.__balances), trades=list(self.__trades))


#  Assuming PriceTakeProfitSuccessExit and TimeoutStopLossFailureExit for now.
class SignalExecutor(BaseSignalExecutor):
    DELTA = 0.0001

    NUM_CHUNKS = EXECUTOR__NUM_CHUNKS
    MIN_CHUNK_SIZE = EXECUTOR__MIN_CHUNK_SIZE

    def __init__(self, market_info, account):
        super().__init__()

        self.market_info = market_info
        self.account = account

        self.__trades = {}

        self.__tracked_signals = {}

    def act(self, signals):
        self.__process_signals()

        market_date = self.__get_market_date()
        estimated_balance = self.account.get_estimated_balance(market_info=self.market_info)
        self.declare_balance(date=market_date, balance=estimated_balance)

        self.__execute_new_signals(trade_signals=signals)

    def __process_signals(self):
        for signal_info in list(self.__tracked_signals.values()):
            self.__process_signal(signal_info=signal_info)

    def __process_signal(self, signal_info):
        open_sell_order_numbers = [order.get_order_number() for order in
                                   self.account.get_open_sell_orders()]
        signal = signal_info['signal']
        sell_order = signal_info['tp_sell_order']
        sell_order_number = sell_order.get_order_number()
        trade_number = sell_order_number
        trade = self.__trades[trade_number]
        market_date = self.__get_market_date()

        if sell_order_number not in open_sell_order_numbers:  # Price TP hit
            print(datetime.fromtimestamp(market_date), sell_order.get_currency(), 'tp')
            close_date = market_date
            profit_amount = trade.amount * trade.target_rate - trade.amount * trade.rate
            self.declare_trade_end(date=close_date,
                                   trade_id=trade_number, profit_amount=profit_amount)
            del self.__trades[trade_number]
            del self.__tracked_signals[signal.id]

        else:
            #  Check Timeout SL
            if market_date - sell_order.get_date() >= signal.failure_exit.timeout:
                print(datetime.fromtimestamp(market_date), sell_order.get_currency(), 'sl')
                self.account.cancel_order(order_number=sell_order_number)
                price = self.market_info.get_pair_ticker(pair=signal.pair).highest_bid
                self.account.sell(currency=sell_order.get_currency(),
                                  price=price, amount=sell_order.get_amount())
                profit_amount = trade.amount * price - trade.amount * trade.rate
                self.declare_trade_end(date=market_date,
                                       trade_id=trade_number, profit_amount=profit_amount)

                del self.__trades[trade_number]
                del self.__tracked_signals[signal.id]

    def __get_pairs_with_open_orders(self):
        return set([pair_from('BTC', order.get_currency()) for order in
                    self.account.get_open_sell_orders()])

    def __get_chunk_size(self, estimated_balance):
        return estimated_balance / self.NUM_CHUNKS

    def __execute_new_signals(self, trade_signals: Iterable[TradeSignal]):
        for trade_signal in trade_signals:
            estimated_balance = self.account.get_estimated_balance(market_info=self.market_info)
            if self.__can_execute_signal(trade_signal=trade_signal,
                                         estimated_balance=estimated_balance):
                position_size = self.__get_chunk_size(estimated_balance=estimated_balance)
                self.__execute_signal(signal=trade_signal, position_size=position_size)

    def __can_execute_signal(self, trade_signal, estimated_balance):
        market_date = self.__get_market_date()
        trade_signal_is_valid = self.__trade_signal_is_valid(trade_signal=trade_signal,
                                                             market_date=market_date)
        no_open_trades_with_pair = self.__no_open_trades_with_pair(trade_signal.pair)
        btc_balance_is_enough = self.__btc_balance_is_enough(estimated_balance=estimated_balance)
        return trade_signal_is_valid and btc_balance_is_enough and no_open_trades_with_pair

    def __get_market_date(self):
        return self.market_info.get_market_time()

    @staticmethod
    def __trade_signal_is_valid(trade_signal, market_date):
        return trade_signal and (market_date - trade_signal.date) < trade_signal.good_for

    def __btc_balance_is_enough(self, estimated_balance):
        chunk_size = self.__get_chunk_size(estimated_balance=estimated_balance)
        chunk_size_is_large_enough = chunk_size >= self.MIN_CHUNK_SIZE
        btc_balance = self.account.get_balance('BTC')
        btc_balance_is_enough = btc_balance >= chunk_size * (1.0 + self.DELTA)
        return chunk_size_is_large_enough and btc_balance_is_enough

    def __no_open_trades_with_pair(self, pair):
        return pair_second(pair) not in \
               [order.get_currency() for order in self.account.get_open_sell_orders()]

    def __execute_signal(self, signal: TradeSignal, position_size):
        entry_price = signal.entry.price
        target_price = signal.success_exit.price
        pair = signal.pair
        market_date = self.__get_market_date()
        currency = pair_second(pair)

        self.account.buy(currency=currency, price=entry_price, amount=position_size / entry_price)

        bought_amount = self.account.get_balance(currency)

        sell_order = Order(currency=currency, _type=OrderType.SELL, price=target_price,
                           amount=bought_amount, date=self.__get_market_date())

        trade_number = sell_order.get_order_number()

        self.account.new_order(order=sell_order)

        self.__save_signal_to_tracked_signals_with_tp_sell_order(signal=signal,
                                                                 tp_sell_order=sell_order)

        self.__trades[trade_number] = self.InternalTrade(currency=currency, amount=bought_amount,
                                                         rate=entry_price,
                                                         target_rate=target_price)

        self.declare_trade_start(date=market_date, trade_number=trade_number)

        self.__print_trade(pair=pair)

    def __save_signal_to_tracked_signals_with_tp_sell_order(self, signal, tp_sell_order):
        self.__tracked_signals[signal.id] = {
            'signal': signal, 'tp_sell_order': tp_sell_order
        }

    def __print_trade(self, pair):
        estimated_balance = self.account.get_estimated_balance(market_info=self.market_info)

        print('TRADE:', pair)
        print('balance:', estimated_balance)
        print('open orders:', len(list(self.account.get_open_sell_orders())))

    class InternalTrade:
        def __init__(self, currency, amount, rate, target_rate):
            self.currency = currency
            self.amount = amount
            self.rate = rate
            self.target_rate = target_rate
