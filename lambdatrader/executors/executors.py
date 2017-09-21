from datetime import datetime
from logging import ERROR
from typing import Iterable

from lambdatrader.config import (
    EXECUTOR__NUM_CHUNKS,
    EXECUTOR__MIN_CHUNK_SIZE,
)
from lambdatrader.models.ordertype import OrderType
from lambdatrader.models.trade import Trade
from lambdatrader.models.tradesignal import TradeSignal
from lambdatrader.models.tradinginfo import TradingInfo
from lambdatrader.utils import pair_from, pair_second
from lambdatrader.loghandlers import get_logger_with_all_handlers, get_logger_with_console_handler, get_silent_logger
from lambdatrader.models.orderrequest import OrderRequest


class BaseSignalExecutor:
    def __init__(self, market_info, account, live=False, silent=False):
        self.market_info = market_info
        self.account = account

        self.LIVE = live
        self.SILENT = silent

        if self.LIVE:
            self.logger = get_logger_with_all_handlers(__name__)
        elif self.SILENT:
            self.logger = get_silent_logger(__name__)
        else:
            self.logger = get_logger_with_console_handler(__name__)
            self.logger.setLevel(ERROR)

        self.__trade_starts = {}
        self.__trades = []
        self.__history_start = None
        self.__history_end = None
        self.__estimated_balances = {}
        self.__frozen_balances = {}
        self.__latest_frozen_balance = None

    def set_history_start(self, date):
        self.__history_start = date

    def set_history_end(self, date):
        self.__history_end = date

    def declare_trade_start(self, date, trade_id):
        self.__trade_starts[trade_id] = date

    def declare_trade_end(self, date, trade_id, profit_amount):
        start_date = self.__trade_starts[trade_id]
        end_date = date
        trade = Trade(_id=trade_id, start_date=start_date, end_date=end_date, profit=profit_amount)
        self.__trades.append(trade)
        self.__set_frozen_balance(date=date, balance=self.get_frozen_balance() + profit_amount)

    def declare_estimated_balance(self, date, balance):
        self.__estimated_balances[date] = balance
        if self.__latest_frozen_balance is None:
            self.__set_frozen_balance(date=date, balance=balance)

    def get_frozen_balance(self):
        return self.__latest_frozen_balance

    def __set_frozen_balance(self, date, balance):
        self.__frozen_balances[date] = balance
        self.__latest_frozen_balance = balance

    def get_trading_info(self):
        return TradingInfo(history_start=self.__history_start, history_end=self.__history_end,
                           estimated_balances=dict(self.__estimated_balances),
                           frozen_balances=dict(self.__frozen_balances),
                           trades=list(self.__trades))

    def act(self, signals):
        raise NotImplementedError


#  Assuming PriceTakeProfitSuccessExit and TimeoutStopLossFailureExit for now.
class SignalExecutor(BaseSignalExecutor):
    DELTA = 0.0001

    NUM_CHUNKS = EXECUTOR__NUM_CHUNKS
    MIN_CHUNK_SIZE = EXECUTOR__MIN_CHUNK_SIZE

    def __init__(self, market_info, account, live=False, silent=False):
        super().__init__(market_info=market_info, account=account, live=live, silent=silent)

        self.__internal_trades = {}

        self.__tracked_signals = {}

    def act(self, signals):
        self.__process_signals()

        market_date = self.__get_market_date()
        estimated_balance = self.account.get_estimated_balance()
        self.declare_estimated_balance(date=market_date, balance=estimated_balance)

        self.__execute_new_signals(trade_signals=signals)

    def __process_signals(self):
        for signal_info in list(self.__tracked_signals.values()):
            self.__process_signal(signal_info=signal_info)

    def __process_signal(self, signal_info):
        open_sell_orders_dict = self.account.get_open_sell_orders()
        signal = signal_info['signal']
        sell_order = signal_info['tp_sell_order']
        sell_order_number = sell_order.get_order_number()
        internal_trade_id = signal.id
        internal_trade = self.__internal_trades[internal_trade_id]
        market_date = self.__get_market_date()

        if sell_order_number not in open_sell_orders_dict:  # Price TP hit
            self.__conditional_print(datetime.fromtimestamp(market_date), sell_order.get_currency(), 'tp')
            close_date = market_date
            profit_amount = self.__calc_profit_amount(amount=internal_trade.amount, buy_rate=internal_trade.rate,
                                                      sell_rate=internal_trade.target_rate)
            self.declare_trade_end(date=close_date,
                                   trade_id=internal_trade_id, profit_amount=profit_amount)
            del self.__internal_trades[internal_trade_id]
            del self.__tracked_signals[signal.id]

        else:
            #  Check Timeout SL
            if market_date - sell_order.get_date() >= signal.failure_exit.timeout:
                self.__conditional_print(datetime.fromtimestamp(market_date), sell_order.get_currency(), 'sl')
                self.account.cancel_order(order_number=sell_order_number)
                price = self.market_info.get_pair_ticker(pair=signal.pair).highest_bid

                sell_request = OrderRequest(currency=sell_order.get_currency(), _type=OrderType.SELL,
                                            price=price, amount=sell_order.get_amount(), date=market_date)

                self.account.new_order(sell_request)
                profit_amount = self.__calc_profit_amount(amount=internal_trade.amount, buy_rate=internal_trade.rate,
                                                          sell_rate=price)
                self.declare_trade_end(date=market_date,
                                       trade_id=internal_trade_id, profit_amount=profit_amount)

                del self.__internal_trades[internal_trade_id]
                del self.__tracked_signals[signal.id]

    def __calc_profit_amount(self, amount, buy_rate, sell_rate):
        bought_amount = amount - self.__get_taker_fee(amount=amount)
        btc_omitted = amount * buy_rate
        btc_added = bought_amount * sell_rate - self.__get_maker_fee(amount=bought_amount * sell_rate)
        return btc_added - btc_omitted

    def __get_taker_fee(self, amount):
        return self.account.get_taker_fee(amount=amount)

    def __get_maker_fee(self, amount):
        return self.account.get_maker_fee(amount=amount)

    def __get_pairs_with_open_orders(self):
        return set([pair_from('BTC', order.get_currency()) for order in
                    self.account.get_open_sell_orders().values()])

    def __get_chunk_size(self, estimated_balance):
        return estimated_balance / self.NUM_CHUNKS

    def __execute_new_signals(self, trade_signals: Iterable[TradeSignal]):
        for trade_signal in trade_signals:
            estimated_balance = self.account.get_estimated_balance()
            if self.__can_execute_signal(trade_signal=trade_signal,
                                         estimated_balance=estimated_balance):
                position_size = self.__get_chunk_size(estimated_balance=estimated_balance)
                self.__execute_signal(signal=trade_signal, position_size=position_size)

    def __can_execute_signal(self, trade_signal, estimated_balance):
        market_date = self.__get_market_date()

        trade_signal_is_valid = self.__trade_signal_is_valid(trade_signal=trade_signal,
                                                             market_date=market_date)
        if not trade_signal_is_valid:
            return False

        no_open_trades_with_pair = self.__no_open_trades_with_pair(trade_signal.pair)
        if not no_open_trades_with_pair:
            return False

        btc_balance_is_enough = self.__btc_balance_is_enough(estimated_balance=estimated_balance)
        if not btc_balance_is_enough:
            return False

        return True

    def __get_market_date(self):
        return self.market_info.get_market_date()

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
               [order.get_currency() for order in self.account.get_open_sell_orders().values()]

    def __execute_signal(self, signal: TradeSignal, position_size):
        entry_price = signal.entry.price
        target_price = signal.success_exit.price
        pair = signal.pair
        currency = pair_second(pair)
        amount_to_buy = position_size / entry_price

        market_date = self.__get_market_date()

        buy_request = OrderRequest(currency=currency, _type=OrderType.BUY,
                                   price=entry_price, amount=amount_to_buy, date=market_date)

        self.account.new_order(order_request=buy_request, fill_or_kill=True)

        bought_amount = self.account.get_balance(currency)

        sell_request = OrderRequest(currency=currency, _type=OrderType.SELL,
                                    price=target_price, amount=bought_amount, date=market_date)

        sell_order = self.account.new_order(order_request=sell_request)

        self.__save_signal_to_tracked_signals_with_tp_sell_order(signal=signal,
                                                                 tp_sell_order=sell_order)

        self.__internal_trades[signal.id] = self.InternalTrade(currency=currency, amount=bought_amount,
                                                               rate=entry_price,
                                                               target_rate=target_price)

        self.declare_trade_start(date=market_date, trade_id=signal.id)

        self.__print_trade(pair=pair)

    def __save_signal_to_tracked_signals_with_tp_sell_order(self, signal, tp_sell_order):
        self.__tracked_signals[signal.id] = {
            'signal': signal, 'tp_sell_order': tp_sell_order
        }

    def __print_trade(self, pair):
        if not self.LIVE and not self.SILENT:
            estimated_balance = self.account.get_estimated_balance()
            frozen_balance = self.get_frozen_balance()

            print()
            print(datetime.fromtimestamp(self.__get_market_date()), 'TRADE:', pair)
            print('estimated_balance:', estimated_balance)
            print('frozen_balance:', frozen_balance)
            print('num_open_orders:', len(list(self.account.get_open_sell_orders())))
            print()

    def __conditional_print(self, *args):
        if not self.LIVE and not self.SILENT:
            print(*args)

    class InternalTrade:
        def __init__(self, currency, amount, rate, target_rate):
            self.currency = currency
            self.amount = amount
            self.rate = rate
            self.target_rate = target_rate
