from enum import Enum
from logging import ERROR
from uuid import uuid1

from account.account import UnableToFillImmediately
from executors.utils import retry_on_exception
from loghandlers import (
    get_logger_with_all_handlers, get_silent_logger, get_logger_with_console_handler,
)
from models.orderrequest import OrderRequest
from models.ordertype import OrderType
from utils import pair_second

FIVE_MINUTES = 5 * 60


class SignalPhase:
    PRE_ENTRY = 1
    IN_PROCESS = 2
    STOP_LOSS = 3
    CLOSED = 4


class EntryType:
    PRICE = 1


class Entry:
    def __init__(self, _type=EntryType.PRICE):
        self.type = _type


class PriceEntry(Entry):
    def __init__(self, price):
        super().__init__(_type=EntryType.PRICE)
        self.price = price


class SuccessExitType(Enum):
    COMBINED_AND = 1
    COMBINED_OR = 2
    PRICE_TAKE_PROFIT = 3


class SuccessExit:
    def __init__(self, _type: SuccessExitType):
        self.type = _type


class CombinedAndSuccessExit(SuccessExit):
    def __init__(self, success_exits):
        super().__init__(_type=SuccessExitType.COMBINED_AND)
        self.success_exits = success_exits[:]


class CombinedOrSuccessExit(SuccessExit):
    def __init__(self, success_exits):
        super().__init__(_type=SuccessExitType.COMBINED_OR)
        self.success_exits = success_exits[:]


class PriceTakeProfitSuccessExit(SuccessExit):
    def __init__(self, price):
        super().__init__(_type=SuccessExitType.PRICE_TAKE_PROFIT)
        self.price = price


class FailureExitType(Enum):
    COMBINED_AND = 1
    COMBINED_OR = 2
    TIMEOUT_STOP_LOSS = 3
    PRICE_STOP_LOSS = 4


class FailureExit:
    def __init__(self, _type: FailureExitType):
        self.type = _type


class CombinedAndFailureExit(FailureExit):
    def __init__(self, failure_exits):
        super().__init__(_type=FailureExitType.COMBINED_AND)
        self.failure_exits = failure_exits[:]


class CombinedOrFailureExit(FailureExit):
    def __init__(self, failure_exits):
        super().__init__(_type=FailureExitType.COMBINED_OR)
        self.failure_exits = failure_exits[:]


class TimeoutStopLossFailureExit(FailureExit):
    def __init__(self, timeout):
        super().__init__(_type=FailureExitType.TIMEOUT_STOP_LOSS)
        self.timeout = timeout


class PriceStopLossFailureExit(FailureExit):
    def __init__(self, price):
        super().__init__(_type=FailureExitType.PRICE_STOP_LOSS)
        self.price = price


class TradeSignal:
    def __init__(self, market, account, date, pair, position_size,
                 entry: PriceEntry,
                 success_exit: PriceTakeProfitSuccessExit,
                 failure_exit: FailureExit,
                 good_for=FIVE_MINUTES,
                 live=True,
                 silent=False,
                 _id=None):
        self.LIVE = live
        self.SILENT = silent

        self.date = date
        self.pair = pair
        self.currency = pair_second(pair)
        self.position_size = position_size
        self.entry = entry
        self.success_exit = success_exit
        self.failure_exit = failure_exit
        self.good_for = good_for

        self.market = market
        self.account = account

        self.current_phase = SignalPhase.PRE_ENTRY
        self.entry_date = None
        self.exit_date = None
        self.entry_rate = None
        self.spent_btc = None
        self.bought_amount = None
        self.sell_order = None
        self.tp_trades = []
        self.sl_trades = []

        if _id:
            self.id = _id
        else:
            self.id = uuid1()

        self._set_up_logger()

    def _set_up_logger(self):
        if self.LIVE:
            self.logger = get_logger_with_all_handlers(__name__)
        else:
            if self.SILENT:
                self.logger = get_silent_logger(__name__)
            else:
                self.logger = get_logger_with_console_handler(__name__)
                self.logger.setLevel(ERROR)

    def __repr__(self):
        return 'TradeSignal(date={},pair={})'.format(self.date, self.pair)

    @property
    def order_size(self):
        return self.spent_btc

    #  Track trade status, do necessary state transitions
    def process(self):
        if self.current_phase is SignalPhase.PRE_ENTRY:
            self._try_entry()
        elif self.current_phase is SignalPhase.IN_PROCESS:
            self._check_in_process_order()
        elif self.current_phase is SignalPhase.STOP_LOSS:
            self._check_stop_loss_order()

    def is_open(self):
        return self.get_current_phase() in [SignalPhase.IN_PROCESS, SignalPhase.STOP_LOSS]

    def is_closed(self):
        return self.get_current_phase() is SignalPhase.CLOSED

    def get_current_phase(self):
        return self.current_phase

    def get_current_profit(self):
        if self.current_phase is SignalPhase.PRE_ENTRY:
            return 0
        else:
            pass  # TODO

    def _try_entry(self):
        btc_balance = self._get_balance_with_retry('BTC')
        estimated_balance = self._get_estimated_balance_with_retry()
        position_size = estimated_balance * self.position_size
        amount_to_buy = position_size / self.entry.price
        market_date = self.market.date

        if btc_balance < position_size:
            return

        try:
            buy_request = OrderRequest(currency=self.currency,
                                       _type=OrderType.BUY,
                                       price=self.entry.price,
                                       amount=amount_to_buy,
                                       date=market_date)
            self._new_order_with_retry(order_request=buy_request, fill_or_kill=True)
            bought_amount = self._get_balance_with_retry(currency=self.currency)
            sell_request = OrderRequest(currency=self.currency,
                                        _type=OrderType.SELL,
                                        price=self.success_exit.price,
                                        amount=bought_amount,
                                        date=market_date)
            sell_order = self._new_order_with_retry(order_request=sell_request)
            self._set_phase_in_process(entry_date=sell_order.get_date(),
                                       entry_rate=self.entry.price,
                                       spent_btc=position_size,
                                       bought_amount=bought_amount,
                                       tp_sell_order=sell_order)
        except UnableToFillImmediately as e:
            self.logger.warning(str(e))

    def _check_in_process_order(self):
        self._process_trades()

        if self.sell_order.number not in self._get_open_sell_orders_with_retry():  # Price TP hit
            self._set_phase_closed(exit_date=self.market.date)

        elif self._failure_exit_satisfied():
                self._cancel_order_with_retry(order_number=self.sell_order.order_number)
                price = self.market.get_pair_ticker(pair=self.pair).highest_bid
                sell_request = OrderRequest(currency=self.sell_order.currency,
                                            _type=OrderType.SELL, price=price,
                                            amount=self.sell_order.amount, date=self.market.date)
                sl_sell_order = self._new_order_with_retry(order_request=sell_request)
                self._set_phase_stop_loss(sl_sell_order=sl_sell_order)

    def _check_stop_loss_order(self):
        open_sell_orders = self._get_open_sell_orders_with_retry()
        highest_bid = self.market.get_pair_ticker(pair=self.pair).highest_bid
        self._process_trades()
        if self.sell_order.order_number not in open_sell_orders:  # SL Filled
            self._set_phase_closed(exit_date=self.market.date)
        else:
            self.sell_order = self._move_order(order=self.sell_order, rate=highest_bid)

    def _process_trades(self):
        pass

    def _set_phase_in_process(self, entry_date, entry_rate,
                              spent_btc, bought_amount, tp_sell_order):
        self.current_phase = SignalPhase.IN_PROCESS
        self.entry_date = entry_date
        self.entry_rate = entry_rate
        self.spent_btc = spent_btc
        self.bought_amount = bought_amount
        self.sell_order = tp_sell_order

    def _set_phase_stop_loss(self, sl_sell_order, tp_trades=[][:]):
        self._add_tp_trade(*tp_trades)
        self.current_phase = SignalPhase.STOP_LOSS
        self.sell_order = sl_sell_order

    def _set_phase_closed(self, exit_date, tp_trades=[][:], sl_trades=[][:]):
        self.exit_date = exit_date
        self._add_tp_trade(*tp_trades)
        self._add_sl_trade(*sl_trades)
        self.current_phase = SignalPhase.CLOSED
        self.sell_order = None

    def _failure_exit_satisfied(self):
        if self.failure_exit.type is FailureExitType.TIMEOUT_STOP_LOSS:
            return self.market.date - self.sell_order.date >= self.failure_exit.timeout

        if self.failure_exit.type is FailureExitType.PRICE_STOP_LOSS:
            highest_bid = self.market.get_pair_ticker(pair=self.pair).highest_bid
            return highest_bid <= self.failure_exit.price

        raise Exception('Unknown or unimplemented failure_exit type.')

    def _move_order(self, order, rate):
        self._cancel_order_with_retry(order_number=order.order_number)
        order_request = OrderRequest(currency=order.currency, _type=order.type, price=order.price, amount=)

    def _unfilled_amount(self, order):
        pass

    def _add_tp_trade(self, *tp_trades):
        for tp_trade in tp_trades:
            self.tp_trades.append(tp_trade)

    def _add_sl_trade(self, *sl_trades):
        for sl_trade in sl_trades:
            self.sl_trades.append(sl_trade)

    def _new_order_with_retry(self, order_request, fill_or_kill=False):
        return self._retry_on_exception(
            lambda: self.account.new_order(order_request=order_request, fill_or_kill=fill_or_kill))

    def _get_balance_with_retry(self, currency):
        return self._retry_on_exception(lambda: self.account.get_balance(currency=currency))

    def _get_order_trades_with_retry(self, order_number):
        return self._retry_on_exception(
            lambda: self.account.get_order_trades(order_number=order_number))

    def _get_open_sell_orders_with_retry(self):
        return self._retry_on_exception(lambda: self.account.get_open_sell_orders())

    def _cancel_order_with_retry(self, order_number):
        return self._retry_on_exception(
            lambda: self.account.cancel_order(order_number=order_number))

    def _get_estimated_balance_with_retry(self):
        return self._retry_on_exception(
            lambda: self.account.get_estimated_balance()
        )

    def _retry_on_exception(self, task, exceptions=None):
        return retry_on_exception(task=task, logger=self.logger, exceptions=exceptions)
