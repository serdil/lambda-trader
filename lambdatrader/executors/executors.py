import pickle
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from logging import ERROR
from threading import Thread, RLock
from typing import Iterable

from lambdatrader.account import (
    UnableToFillImmediately,
)
from lambdatrader.config import (
    EXECUTOR__NUM_CHUNKS, EXECUTOR__MIN_CHUNK_SIZE, BOT_IDENTIFIER,
)
from lambdatrader.evaluation.utils import period_statistics
from lambdatrader.executors.utils import retry_on_exception
from lambdatrader.loghandlers import (
    get_logger_with_all_handlers, get_logger_with_console_handler, get_silent_logger,
)
from lambdatrader.models.orderrequest import OrderRequest
from lambdatrader.models.ordertype import OrderType
from lambdatrader.models.trade import Trade
from lambdatrader.models.tradesignal import TradeSignal, FailureExitType
from lambdatrader.models.tradinginfo import TradingInfo
from lambdatrader.object_persistence import get_object_with_key, save_object_with_key
from lambdatrader.utilities.utils import pair_from, pair_second


class BaseSignalExecutor:
    MEMORY_VERSION = 0

    def __init__(self, market_info, account, live=False, silent=False):
        self.market_info = market_info
        self.account = account

        self.LIVE = live
        self.SILENT = silent

        self.__set_up_logger()
        self.__set_up_memory()
        self.__set_up_memory_fields()

    def __set_up_logger(self):
        if self.LIVE:
            self.logger = get_logger_with_all_handlers(__name__)
        else:
            if self.SILENT:
                self.logger = get_silent_logger(__name__)
            else:
                self.logger = get_logger_with_console_handler(__name__)
                self.logger.setLevel(ERROR)

    def __set_up_memory(self):
        self._memory_lock = RLock()

        with self._memory_lock:
            if self.LIVE:
                self._memory = self._get_memory_from_db()
            else:
                self._memory = self._get_default_memory()

    def __set_up_memory_fields(self):
        if 'trades' not in self._memory_memory:
            self._memory_memory['trades'] = []

        if 'history_start' not in self._memory_memory:
            self._memory_memory['history_start'] = self.market_info.market_date

        if 'history_end' not in self._memory_memory:
            self._memory_memory['history_end'] = self.market_info.market_date

        if 'estimated_balances' not in self._memory_memory:
            self._memory_memory['estimated_balances'] = {}

        if 'frozen_balances' not in self._memory_memory:
            self._memory_memory['frozen_balances'] = {}

        if 'latest_frozen_balance' not in self._memory_memory:
            self._memory_memory['latest_frozen_balance'] = None

    @property
    def _memory_memory(self):
        return self._memory['memory']

    @property
    def __history_start(self):
        return self._memory_memory['history_start']

    @property
    def __history_end(self):
        return self._memory_memory['history_end']

    @property
    def __trades(self):
        return self._memory_memory['trades']

    @property
    def __estimated_balances(self):
        return self._memory_memory['estimated_balances']

    @property
    def __frozen_balances(self):
        return self._memory_memory['frozen_balances']

    @property
    def __latest_frozen_balance(self):
        return self._memory_memory['latest_frozen_balance']

    def _get_memory_from_db(self):
        self.debug('get_memory_from_db')

        memory_dict = get_object_with_key(self._get_memory_key())

        if memory_dict is None:
            self.logger.warning('memory_object_not_found_in_db_creating_new')
            memory_dict = self._get_default_memory()

        if memory_dict['version'] != self.MEMORY_VERSION:
            raise RuntimeError('Memory versions do not match,'
                               ' db:{} code:{}'.format(memory_dict['version'], self.MEMORY_VERSION))

        self.debug('got_memory_from_db')
        return memory_dict

    def _get_default_memory(self):
        return {'version': self.MEMORY_VERSION, 'memory': {}}

    def _save_memory_to_db(self):
        self.debug('save_memory_to_db, key:%s', self._get_memory_key())
        save_object_with_key(self._get_memory_key(), self._memory)

    def _get_memory_key(self):
        return '{}.{}.memory'.format(BOT_IDENTIFIER, self.__class__.__name__)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def set_history_start(self, date):
        if date > self.__history_end:
            raise ValueError('Start date cannot be later than end date.')
        self._memory_memory['history_start'] = date

    def set_history_end(self, date):
        self._memory_memory['history_end'] = date

    def update_history_end_with_current_date(self):
        self.set_history_end(self.market_info.market_date)

    def _declare_trade_end(self, trade):
        self.__trades.append(trade)

        end_date = trade.end_date
        profit_amount = trade.profit_amount
        self.__set_frozen_balance(date=end_date, balance=self.get_frozen_balance() + profit_amount)
        self.update_history_end_with_current_date()

    def declare_estimated_balance(self, date, balance):
        self.__estimated_balances[date] = balance
        self.update_history_end_with_current_date()
        if self.__latest_frozen_balance is None:
            self.__set_frozen_balance(date=date, balance=balance)

    def get_frozen_balance(self):
        return self.__latest_frozen_balance

    def __set_frozen_balance(self, date, balance):
        self.__frozen_balances[date] = balance
        self._memory_memory['latest_frozen_balance'] = balance
        self.update_history_end_with_current_date()

    def get_trading_info(self):
        return TradingInfo(history_start=self.__history_start, history_end=self.__history_end,
                           estimated_balances=dict(self.__estimated_balances),
                           frozen_balances=dict(self.__frozen_balances),
                           trades=list(self.__trades))

    def act(self, signals):
        raise NotImplementedError

    def get_market_date(self):
        return self.market_date

    @property
    def market_date(self):
        return self.market_info.market_date


#  Assuming PriceTakeProfitSuccessExit and TimeoutStopLossFailureExit and PriceStopLossFailureExit
#  for now.
class SignalExecutor(BaseSignalExecutor):
    MEMORY_VERSION = 0

    DELTA = 0.0001

    NUM_CHUNKS = EXECUTOR__NUM_CHUNKS
    MIN_CHUNK_SIZE = EXECUTOR__MIN_CHUNK_SIZE

    def __init__(self, market_info, account, live=False, silent=False):
        super().__init__(market_info=market_info, account=account, live=live, silent=silent)

        self.debug('initializing_signal_executor')

        self.__set_up_memory_fields()
        self.__set_up_scheduled_tasks()

    def __set_up_memory_fields(self):
        if 'internal_trades' not in self._memory_memory:
            self._memory_memory['internal_trades'] = {}

        if 'tracked_signals' not in self._memory_memory:
            self._memory_memory['tracked_signals'] = {}

    def __set_up_scheduled_tasks(self):
        self.__scheduled_tasks = defaultdict(list)

        self.__schedule_task(task=self.__process_signals, time_offset=0, period=5)
        self.__schedule_task(task=self.__report_estimated_balance, time_offset=0, period=60)

        if self.LIVE:
            self.__schedule_task(task=lambda: self.__heartbeat(count=0), time_offset=0)
            self.__schedule_task(task=self.__compute_statistics, time_offset=0, period=21600)

    def __schedule_task(self, task, time_offset, period=None):
        scheduled_time = self.market_date + time_offset
        self.__scheduled_tasks[scheduled_time].append((task, period))

    def __run_scheduled_tasks(self):
        self.debug('__run_scheduled_tasks')

        market_date = self.market_date

        times_to_delete = []
        tasks_to_readd = []
        tasks_to_run = []
        for scheduled_time, tasks in self.__scheduled_tasks.items():
            if market_date >= scheduled_time:
                times_to_delete.append(scheduled_time)
                for task_tuple in tasks:
                    tasks_to_run.append(task_tuple)
                    if task_tuple[1] is not None:
                        tasks_to_readd.append(task_tuple)

        for task_tuple in tasks_to_run:
            try:
                self.debug('__running_task')
                task_tuple[0]()
            except Exception:
                self.logger.exception('exception in scheduled task')

        for time in times_to_delete:
            del self.__scheduled_tasks[time]

        for task_tuple in tasks_to_readd:
            self.debug('__rescheduling_task')
            scheduled_time = market_date + task_tuple[1]
            self.__scheduled_tasks[scheduled_time].append(task_tuple)

    def __save_memory(self):
        self.debug('__save_memory')
        self._save_memory_to_db()

    @property
    def __internal_trades(self):
        return self._memory_memory['internal_trades']

    @property
    def __tracked_signals(self):
        return self._memory_memory['tracked_signals']

    def act(self, signals):
        with self._memory_lock:
            self.debug('act')

            self.__run_scheduled_tasks()

            self.__execute_new_signals(trade_signals=signals)

            if self.LIVE:
                self.__save_memory()

            tracked_signals_list = self.__get_tracked_signals_list()

            self.debug('end_of_act')
            return tracked_signals_list

    def __report_estimated_balance(self):
        market_date = self.market_date
        estimated_balance = self.__get_estimated_balance_with_retry()
        self.declare_estimated_balance(date=market_date, balance=estimated_balance)

    def __get_estimated_balance_with_retry(self):
        return self.retry_on_exception(
            lambda: self.account.get_estimated_balance()
        )

    def __process_signals(self):
        self.debug('__process_signals')
        self.debug('__tracked_signals:%s', self.__tracked_signals)
        for signal_info in list(self.__tracked_signals.values()):
            self.__process_signal(signal_info=signal_info)

    def __process_signal(self, signal_info):
        self.debug('__process_signal')

        open_sell_orders_dict = self.__get_open_sell_orders_with_retry()
        signal = signal_info['signal']
        sell_order = signal_info['tp_sell_order']
        sell_order_number = sell_order.get_order_number()
        internal_trade_id = signal.id
        internal_trade = self.__internal_trades[internal_trade_id]
        market_date = self.market_date

        in_stop_loss_stage = 'in_stop_loss_stage' in signal_info and\
                             signal_info['in_stop_loss_stage']

        # Price TP hit
        if not in_stop_loss_stage and sell_order_number not in open_sell_orders_dict:
            self.logger.info('tp_hit_for_signal:%s', signal)

            profit_amount = self.__calc_profit_amount(amount=internal_trade.amount,
                                                      buy_rate=internal_trade.rate,
                                                      sell_rate=internal_trade.target_rate)

            trade = Trade(_id=internal_trade_id, start_date=internal_trade.start_date,
                          end_date=market_date, profit=profit_amount)
            self._declare_trade_end(trade=trade)

            self.logger.info('trade_closed_p_l:%.6f', profit_amount)
            self.__print_tp_hit_for_backtesting(market_date=market_date,
                                                currency=sell_order.get_currency(),
                                                profit_amount=profit_amount)

            del self.__internal_trades[internal_trade_id]
            del self.__tracked_signals[signal.id]

        else:
            failure_exit = signal.failure_exit
            if failure_exit.type == FailureExitType.TIMEOUT_STOP_LOSS:
                stop_loss_reached = market_date - sell_order.get_date() >= failure_exit.timeout
            elif failure_exit.type == FailureExitType.PRICE_STOP_LOSS:
                highest_bid = self.market_info.get_pair_ticker(pair=signal.pair).highest_bid
                stop_loss_reached = highest_bid <= failure_exit.price
            else:
                raise Exception('Unknown or unimplemented failure_exit type.')

            if stop_loss_reached:
                self.logger.info('sl_hit_for_signal:%s', signal)
                self.logger.info('cancelling_order:%s;', sell_order)

                if not in_stop_loss_stage:
                    self.__cancel_order_with_retry(order_number=sell_order_number)
                signal_info['in_stop_loss_stage'] = True
                amount_to_sell = self.__get_balance_with_retry(sell_order.get_currency())
                price = self.market_info.get_pair_ticker(pair=signal.pair).highest_bid

                self.logger.info('trying_to_sell_at_current_price')
                sell_request = OrderRequest(currency=sell_order.get_currency(),
                                            _type=OrderType.SELL, price=price,
                                            amount=amount_to_sell,
                                            date=market_date)
                try:
                    self.__new_order_with_retry(order_request=sell_request, fill_or_kill=True)
                except UnableToFillImmediately as e:
                    self.logger.warning(str(e))
                    return

                profit_amount = self.__calc_profit_amount(amount=internal_trade.amount,
                                                          buy_rate=internal_trade.rate,
                                                          sell_rate=price)

                trade = Trade(_id=internal_trade_id, start_date=internal_trade.start_date,
                              end_date=market_date, profit=profit_amount)
                self._declare_trade_end(trade=trade)

                self.logger.info('trade_closed_p_l:%.6f', profit_amount)
                self.__print_sl_hit_for_backtesting(market_date=market_date,
                                                    currency=sell_order.get_currency(),
                                                    profit_amount=profit_amount)

                del self.__internal_trades[internal_trade_id]
                del self.__tracked_signals[signal.id]

    def __get_open_sell_orders_with_retry(self):
        return self.retry_on_exception(
            lambda: self.account.get_open_sell_orders()
        )

    def __cancel_order_with_retry(self, order_number):
        return self.retry_on_exception(
            lambda: self.account.cancel_order(order_number=order_number)
        )

    def __print_tp_hit_for_backtesting(self, market_date, currency, profit_amount):
        self.__conditional_print(datetime.utcfromtimestamp(market_date),
                                 currency, 'tp:', profit_amount)

    def __print_sl_hit_for_backtesting(self, market_date, currency, profit_amount):
        self.__conditional_print(datetime.utcfromtimestamp(market_date),
                                 currency, 'sl:', profit_amount)

    def __calc_profit_amount(self, amount, buy_rate, sell_rate, sell_is_fill_or_kill=True):
        bought_amount = amount - self.__get_taker_fee(amount=amount)
        btc_omitted = amount * buy_rate
        if sell_is_fill_or_kill:
            btc_added = (bought_amount * sell_rate -
                         self.__get_taker_fee(amount=bought_amount * sell_rate))
        else:
            btc_added = (bought_amount * sell_rate -
                         self.__get_maker_fee(amount=bought_amount * sell_rate))
        return btc_added - btc_omitted

    def __get_taker_fee(self, amount):
        return self.account.get_taker_fee(amount=amount)

    def __get_maker_fee(self, amount):
        return self.account.get_maker_fee(amount=amount)

    def __get_chunk_size(self, estimated_balance):
        return estimated_balance / self.NUM_CHUNKS

    def __execute_new_signals(self, trade_signals: Iterable[TradeSignal]):
        for trade_signal in trade_signals:
            estimated_balance = self.__get_estimated_balance_with_retry()
            if self.__can_execute_signal(trade_signal=trade_signal,
                                         estimated_balance=estimated_balance):
                position_size = self.__get_chunk_size(estimated_balance=estimated_balance)
                self.__execute_signal(signal=trade_signal, position_size=position_size)

    def __can_execute_signal(self, trade_signal, estimated_balance):
        market_date = self.market_date

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

    @staticmethod
    def __trade_signal_is_valid(trade_signal, market_date):
        return trade_signal and (market_date - trade_signal.date) < trade_signal.good_for

    def __btc_balance_is_enough(self, estimated_balance):
        chunk_size = self.__get_chunk_size(estimated_balance=estimated_balance)
        chunk_size_is_large_enough = chunk_size >= self.MIN_CHUNK_SIZE
        btc_balance = self.__get_balance_with_retry('BTC')
        btc_balance_is_enough = btc_balance >= chunk_size * (1.0 + self.DELTA)
        return chunk_size_is_large_enough and btc_balance_is_enough

    def __no_open_trades_with_pair(self, pair):
        return pair_second(pair) not in \
               [order.get_currency() for order in self.__get_open_sell_orders_with_retry().values()]

    def __execute_signal(self, signal: TradeSignal, position_size):
        self.logger.info('executing_signal:%s', signal)

        entry_price = signal.entry.price
        target_price = signal.success_exit.price
        pair = signal.pair
        currency = pair_second(pair)
        amount_to_buy = position_size / entry_price

        market_date = self.market_date

        buy_request = OrderRequest(currency=currency, _type=OrderType.BUY,
                                   price=entry_price, amount=amount_to_buy, date=market_date)

        try:
            self.__new_order_with_retry(order_request=buy_request, fill_or_kill=True)
        except UnableToFillImmediately as e:
            self.logger.warning(str(e))
            return

        bought_amount = self.__get_balance_with_retry(currency=currency)

        sell_request = OrderRequest(currency=currency, _type=OrderType.SELL,
                                    price=target_price, amount=bought_amount, date=market_date)

        sell_order = self.__new_order_with_retry(order_request=sell_request)

        self.__save_signal_to_tracked_signals_with_tp_sell_order(signal=signal,
                                                                 tp_sell_order=sell_order)

        self.__internal_trades[signal.id] = self.InternalTrade(start_date=market_date,
                                                               currency=currency,
                                                               amount=bought_amount,
                                                               rate=entry_price,
                                                               target_rate=target_price)

        self.__print_trade_for_backtesting(pair=pair)

    def __new_order_with_retry(self, order_request, fill_or_kill=False):
        return self.retry_on_exception(
            lambda: self.account.new_order(order_request=order_request, fill_or_kill=fill_or_kill)
        )

    def __get_balance_with_retry(self, currency):
        return self.retry_on_exception(
            lambda: self.account.get_balance(currency=currency)
        )

    def __save_signal_to_tracked_signals_with_tp_sell_order(self, signal, tp_sell_order):
        self.__tracked_signals[signal.id] = {
            'signal': signal, 'tp_sell_order': tp_sell_order
        }

    def __get_tracked_signals_list(self):
        tracked_signals = []
        for tracked_signal_info in self.__tracked_signals.values():
            tracked_signals.append(tracked_signal_info['signal'])
        return tracked_signals

    def retry_on_exception(self, task, exceptions=None):
        return retry_on_exception(task=task, logger=self.logger, exceptions=exceptions)

    def __print_trade_for_backtesting(self, pair):
        if self.__in_non_silent_backtesting():
            estimated_balance = self.account.get_estimated_balance()
            frozen_balance = self.get_frozen_balance()

            self.__conditional_print()
            self.__conditional_print(datetime.utcfromtimestamp(self.market_date),
                                     'TRADE:', pair)
            self.__conditional_print('estimated_balance:', estimated_balance)
            self.__conditional_print('frozen_balance:', frozen_balance)
            self.__conditional_print('num_open_orders:',
                                     len(list(self.account.get_open_sell_orders())))
            self.__conditional_print()

    def __conditional_print(self, *args):
        if self.__in_non_silent_backtesting():
            print(*args)

    def __in_non_silent_backtesting(self):
        return not self.LIVE and not self.SILENT

    def __heartbeat(self, count):
        if self.LIVE:
            with self._memory_lock:
                try:
                    if count % 4 == 0:
                        self.__log_heartbeat_info_conditionally(log_if_no_open_orders=True)
                    else:
                        self.__log_heartbeat_info_conditionally(log_if_no_open_orders=False)
                except Exception:
                    self.logger.exception('exception in heartbeat')
                self.__schedule_task(task=lambda: self.__heartbeat(count+1), time_offset=1800)

    def __log_heartbeat_info_conditionally(self, log_if_no_open_orders=False):
        trades = self.__internal_trades.values()

        if len(trades) == 0 and not log_if_no_open_orders:
            return

        trades_p_l = {}

        for trade in trades:
            pair = pair_from('BTC', trade.currency)
            highest_bid = self.market_info.get_pair_ticker(pair=pair).highest_bid
            trades_p_l[pair] = self.__calc_profit_amount(amount=trade.amount,
                                                         buy_rate=trade.rate,
                                                         sell_rate=highest_bid)

        estimated_balance = self.__get_estimated_balance_with_retry()
        frozen_balance = self.get_frozen_balance()
        self.__log_heartbeat_info(frozen_balance=frozen_balance,
                                  estimated_balance=estimated_balance,
                                  trades_p_l=trades_p_l)

    def __log_heartbeat_info(self, frozen_balance, estimated_balance, trades_p_l):
        num_open_orders = len(trades_p_l)
        p_l_summary = self.__get_p_l_summary_string(trades_p_l=trades_p_l)
        if p_l_summary == '':
            self.logger.info('HEARTBEAT: frozen_balance:%f estimated_balance:%f num_open_orders:%d',
                             frozen_balance, estimated_balance, num_open_orders)
        else:
            self.logger.info('HEARTBEAT: frozen_balance:%f estimated_balance:%f num_open_orders:%d'
                             ' p/l summary: %s',
                             frozen_balance, estimated_balance, num_open_orders, p_l_summary)

    @staticmethod
    def __get_p_l_summary_string(trades_p_l):
        return ','.join(['{}:{:.6f}'.format(item[0], item[1]) for item in trades_p_l.items()])

    def __copy_internal_trades(self):
        with self._memory_lock:
            internal_trades = self.__internal_trades
            internal_trades_copy = deepcopy(internal_trades)
            return internal_trades_copy

    def __compute_statistics(self):
        trading_info = self.get_trading_info()
        self.logger.info('trading_info size:%f MB', self.__compute_object_size(trading_info))
        statistics = period_statistics(trading_info=trading_info, end=self.market_date)
        self.logger.info('overall_statistics:%s', statistics)

    @staticmethod
    def __compute_object_size(obj):
        return len(pickle.dumps(obj, protocol=4)) / 1024 / 1024

    @staticmethod
    def __run_in_separate_thread(task):
        Thread(task).run()

    class InternalTrade:
        def __init__(self, start_date, currency, amount, rate, target_rate):
            self.start_date = start_date
            self.currency = currency
            self.amount = amount
            self.rate = rate
            self.target_rate = target_rate
