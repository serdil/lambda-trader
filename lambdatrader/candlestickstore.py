import os
import sqlite3
from collections import defaultdict
from threading import RLock

from blist import sorteddict

from lambdatrader.config import CANDLESTICK_DB_DIRECTORY
from lambdatrader.constants import PeriodEnum, M5
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.models.candlestick import Candlestick
from lambdatrader.utilities.utils import date_floor, date_ceil, seconds

DATABASE_DIR = CANDLESTICK_DB_DIRECTORY

if not os.path.isdir(DATABASE_DIR):
    os.makedirs(DATABASE_DIR, exist_ok=True)


def rlock(method):
    def wrapper(*args, **kwargs):
        if not hasattr(args[0], '_access_lock'):
            args[0]._access_lock = RLock()
        with args[0]._access_lock:
            return method(*args, **kwargs)
    return wrapper


class CandlestickStore:  # TODO make thread safe

    class __CandlestickStore:

        ONE_CHUNK_SECONDS = seconds(days=2)

        def __init__(self, exchange: ExchangeEnum):
            self.__conn = self.__get_db_connection(exchange=exchange)
            self.__cursor = self.__conn.cursor()

            self.__history = defaultdict(sorteddict)
            self.__chunks_in_memory = defaultdict(set)
            self.__chunks_in_db = defaultdict(set)
            self.__synced_pair_periods = set()

            self.__sync_with_existing_pairs_in_db()

        @rlock
        def get_pairs(self):
            return list(
                map(self.pair_from_pair_period,
                    set(self.__history.keys()) | set(self.__chunks_in_memory.keys())
                    | set(self.__chunks_in_db.keys()) | self.__synced_pair_periods)
            )

        @rlock
        def append_candlestick(self, pair, candlestick: Candlestick):
            period = candlestick.period
            pair_period = self.pair_period_name(pair, candlestick.period)

            self.__sync_pair_period_if_not_synced(pair_period)
            newest_date = self.get_pair_period_newest_date(pair=pair, period=period)

            if newest_date is not None and candlestick.date > newest_date + period.seconds():
                error_message = 'Candlestick date {} is not an increment of latest date {}'
                raise ValueError(error_message.format(candlestick.date, newest_date))

            self.__history[pair_period][candlestick.date] = candlestick
            self.__chunks_in_memory[pair_period].add(self.__get_chunk_no(date=candlestick.date))

        @rlock
        def get_candlestick(self, pair, date, period=M5):
            pair_period = self.pair_period_name(pair, period)
            self.__sync_pair_period_if_not_synced(pair_period)
            if self.__get_chunk_no(date=date) not in self.__chunks_in_memory[pair_period]:
                self.__load_chunk(pair_period=pair_period, chunk_no=self.__get_chunk_no(date=date))
            return self.__history[pair_period][date]

        @rlock
        def get_pair_period_oldest_date(self, pair, period=M5):
            pair_period = self.pair_period_name(pair, period)
            self.__sync_pair_period_if_not_synced(pair_period=pair_period)
            pair_period_history = self.__history[pair_period]

            if len(pair_period_history.keys()) == 0:
                return None

            return pair_period_history[pair_period_history.keys()[0]].date

        @rlock
        def get_pair_period_newest_date(self, pair, period=M5):
            pair_period = self.pair_period_name(pair, period)
            self.__sync_pair_period_if_not_synced(pair_period=pair_period)
            pair_period_history = self.__history[pair_period]

            if len(pair_period_history.keys()) == 0:
                return None

            return pair_period_history[pair_period_history.keys()[-1]].date

        def __sync_with_existing_pairs_in_db(self):
            for pair_period_table_name in self.__get_pair_period_table_names():
                self.__sync_pair_period_if_not_synced(pair_period=pair_period_table_name)

        def __get_pair_period_table_names(self):
            query = "SELECT name FROM sqlite_master WHERE type='table'"
            return [
                row[0] for row in self.__cursor.execute(query).fetchall()
                if row[0].find('BTC') == 0
            ]

        def __get_chunk_no(self, date):
            return date // self.ONE_CHUNK_SECONDS

        def __sync_pair_period_if_not_synced(self, pair_period):
            if pair_period not in self.__synced_pair_periods:
                self.__sync_pair_period(pair_period=pair_period)

        def __sync_pair_period(self, pair_period):
            self.__create_pair_period_table_if_not_exists(pair_period=pair_period)

            if self.__pair_period_table_is_empty(pair_period):
                return

            self.__load_pair_period_first_chunk(pair_period=pair_period)
            self.__load_pair_period_last_chunk(pair_period=pair_period)

            self.__synced_pair_periods.add(pair_period)

        def __load_pair_period_first_chunk(self, pair_period):
            self.__load_chunk(pair_period,
                              self.__get_chunk_no(
                                  date=self.__get_pair_period_oldest_date_from_db(pair_period)
                              ),
                              mark_as_chunk_in_db=False)

        def __load_pair_period_last_chunk(self, pair_period):
            self.__load_chunk(pair_period,
                              self.__get_chunk_no(
                                  date=self.__get_pair_period_newest_date_from_db(pair_period)
                              ),
                              mark_as_chunk_in_db=False)

        def __pair_period_table_is_empty(self, pair_period):
            query = "SELECT * FROM '{}' LIMIT 1".format(pair_period)
            return len(self.__cursor.execute(query).fetchall()) == 0

        def __get_pair_period_oldest_date_from_db(self, pair_period):
            query = "SELECT date FROM '{}' ORDER BY date ASC LIMIT 1".format(pair_period)
            return self.__cursor.execute(query).fetchone()[0]

        def __get_pair_period_newest_date_from_db(self, pair_period):
            query = "SELECT date FROM '{}' ORDER BY date DESC LIMIT 1".format(pair_period)
            return self.__cursor.execute(query).fetchone()[0]

        def __load_chunk(self, pair_period, chunk_no, mark_as_chunk_in_db=True):
            period = self.period_from_pair_period(pair_period)
            chunk_start_date = chunk_no * self.ONE_CHUNK_SECONDS
            chunk_end_date = chunk_start_date + self.ONE_CHUNK_SECONDS

            query = "SELECT * FROM '{}' WHERE date >= ? AND date < ? ORDER BY date ASC"\
                .format(pair_period)
            self.__cursor.execute(query, (chunk_start_date, chunk_end_date,))

            for row in self.__cursor:
                candlestick = self.__make_candlestick_from_row(row=row, period=period)
                self.__history[pair_period][candlestick.date] = candlestick

            if mark_as_chunk_in_db:
                self.__chunks_in_db[pair_period].add(chunk_no)
            self.__chunks_in_memory[pair_period].add(chunk_no)

        @rlock
        def persist_chunks(self):
            for pair_period, chunks_in_memory in self.__chunks_in_memory.items():
                pair = self.pair_from_pair_period(pair_period)
                period = self.period_from_pair_period(pair_period)
                for i, chunk_no in enumerate(sorted(chunks_in_memory)):
                    if chunk_no not in self.__chunks_in_db[pair_period]:
                        if i == 0:
                            start_from = self.get_pair_period_oldest_date(pair=pair,
                                                                          period=period) \
                                         % self.ONE_CHUNK_SECONDS
                            self.__persist_chunk(pair_period=pair_period,
                                                 chunk_no=chunk_no,
                                                 start_offset=start_from)
                        else:
                            self.__persist_chunk(pair_period=pair_period, chunk_no=chunk_no)

        def __persist_chunk(self, pair_period, chunk_no, start_offset=0):
            period = self.period_from_pair_period(pair_period)
            rows_to_insert = []
            start_date = chunk_no * self.ONE_CHUNK_SECONDS + start_offset
            for i in range(start_date, start_date + self.ONE_CHUNK_SECONDS, period.seconds()):
                try:
                    candlestick = self.__history[pair_period][i]
                    rows_to_insert.append(self.__make_row_from_candlestick(candlestick))
                except KeyError:
                    break

            self.__cursor.executemany(
                "INSERT OR REPLACE INTO '{}' VALUES (?, ?, ?, ?, ?, ?, ?, ?)".format(pair_period),
                rows_to_insert
            )
            self.__conn.commit()

        @staticmethod
        def __make_candlestick_from_row(row, period):
            return Candlestick(period=period, date=row[0], high=row[1], low=row[2],
                               _open=row[3], close=row[4], base_volume=row[5],
                               quote_volume=row[6], weighted_average=row[7])

        @staticmethod
        def __make_row_from_candlestick(candlestick):
            return (candlestick.date, candlestick.high, candlestick.low, candlestick.open,
                    candlestick.close, candlestick.base_volume, candlestick.quote_volume,
                    candlestick.weighted_average)

        @staticmethod
        def __date_floor(date):
            return date_floor(date)

        @staticmethod
        def __date_ceil(date):
            return date_ceil(date)

        @staticmethod
        def __get_db_connection(exchange: ExchangeEnum):
            db_path = os.path.join(DATABASE_DIR, '{}.db'.format(exchange.name))
            return sqlite3.connect(db_path)

        def __create_pair_period_table_if_not_exists(self, pair_period):
            self.__cursor.execute('''CREATE TABLE IF NOT EXISTS '{}'
                                    (date INTEGER PRIMARY KEY ASC, high REAL, low REAL, 
                                    open REAL, close REAL, base_volume REAL,
                                    quote_volume REAL, weighted_average REAL)'''
                                  .format(pair_period))
            self.__conn.commit()

        @staticmethod
        def pair_period_name(pair, period: PeriodEnum):
            return '{}:{}'.format(pair, period.name)

        @staticmethod
        def pair_from_pair_period(pair_period):
            return pair_period[:pair_period.index(':')]

        @staticmethod
        def period_from_pair_period(pair_period):
            return PeriodEnum.from_name(pair_period[pair_period.index(':')+1:])

    __instances = {}

    @rlock
    @classmethod
    def get_for_exchange(cls, exchange: ExchangeEnum=ExchangeEnum.POLONIEX):
        if exchange not in cls.__instances:
            cls.__instances[exchange] = cls.__CandlestickStore(exchange=exchange)
        return cls.__instances[exchange]
