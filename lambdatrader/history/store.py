import sqlite3

from collections import defaultdict

from blist import sorteddict

from lambdatrader.config import HISTORY_DB_PATH
from lambdatrader.models.candlestick import Candlestick
from lambdatrader.utils import date_floor, date_ceil

DATABASE_PATH = HISTORY_DB_PATH


class CandlestickStore:  # TODO make thread safe

    class __CandlestickStore:

        ONE_CHUNK_SECONDS = 86400

        def __init__(self):
            self.__conn = self.__get_db_connection()
            self.__cursor = self.__conn.cursor()

            self.__history = defaultdict(sorteddict)
            self.__chunks_in_memory = defaultdict(set)
            self.__chunks_in_db = defaultdict(set)
            self.__synced_pairs = set()

            self.__sync_with_existing_pairs_in_db()

        def get_pairs(self):
            return list(filter(lambda pair: pair not in ['BTC_NOTE', 'BTC_SJCX', 'BTC_NAUT'],
                               list(set(self.__history.keys()) | set(self.__chunks_in_memory.keys())
                                    | set(self.__chunks_in_db.keys()) | self.__synced_pairs)))

        def append_candlestick(self, pair, candlestick: Candlestick):
            self.__sync_pair_if_not_synced(pair)
            newest_date = self.get_pair_newest_date(pair)

            if newest_date is not None and candlestick.date > newest_date + 300:
                error_message = 'Candlestick date {} is not an increment of latest date {}'
                raise ValueError(error_message.format(candlestick.date, newest_date))

            self.__history[pair][candlestick.date] = candlestick
            self.__chunks_in_memory[pair].add(self.__get_chunk_no(date=candlestick.date))

        def get_candlestick(self, pair, date):
            self.__sync_pair_if_not_synced(pair=pair)
            if self.__get_chunk_no(date=date) not in self.__chunks_in_memory[pair]:
                self.__load_chunk(pair=pair, chunk_no=self.__get_chunk_no(date=date))
            return self.__history[pair][date]

        def get_pair_oldest_date(self, pair):
            self.__sync_pair_if_not_synced(pair=pair)
            pair_history = self.__history[pair]

            if len(pair_history.keys()) == 0:
                return None

            return pair_history[pair_history.keys()[0]].date

        def get_pair_newest_date(self, pair):
            self.__sync_pair_if_not_synced(pair=pair)
            pair_history = self.__history[pair]

            if len(pair_history.keys()) == 0:
                return None

            return pair_history[pair_history.keys()[-1]].date

        def __sync_with_existing_pairs_in_db(self):
            for pair_table_name in self.__get_pair_table_names():
                self.__sync_pair_if_not_synced(pair=pair_table_name)

        def __get_pair_table_names(self):
            query = "SELECT name FROM sqlite_master WHERE type='table'"
            return [
                row[0] for row in self.__cursor.execute(query).fetchall() if row[0].find('BTC') == 0
            ]

        def __get_chunk_no(self, date):
            return date // self.ONE_CHUNK_SECONDS

        def __sync_pair_if_not_synced(self, pair):
            if pair not in self.__synced_pairs:
                self.__sync_pair(pair=pair)

        def __sync_pair(self, pair):
            self.__create_pair_table_if_not_exists(pair=pair)

            if self.__pair_table_is_empty(pair):
                return

            self.__load_pair_first_chunk(pair=pair)
            self.__load_pair_last_chunk(pair=pair)

            self.__synced_pairs.add(pair)

        def __load_pair_first_chunk(self, pair):
            self.__load_chunk(pair,
                              self.__get_chunk_no(date=self.__get_pair_oldest_date_from_db(pair)),
                              mark_as_chunk_in_db=False)

        def __load_pair_last_chunk(self, pair):
            self.__load_chunk(pair,
                              self.__get_chunk_no(date=self.__get_pair_newest_date_from_db(pair)),
                              mark_as_chunk_in_db=False)

        def __pair_table_is_empty(self, pair):
            query = 'SELECT * FROM {} LIMIT 1'.format(pair)
            return len(self.__cursor.execute(query).fetchall()) == 0

        def __get_pair_oldest_date_from_db(self, pair):
            query = 'SELECT date FROM {} ORDER BY date ASC LIMIT 1'.format(pair)
            return self.__cursor.execute(query).fetchone()[0]

        def __get_pair_newest_date_from_db(self, pair):
            query = 'SELECT date FROM {} ORDER BY date DESC LIMIT 1'.format(pair)
            return self.__cursor.execute(query).fetchone()[0]

        def __load_chunk(self, pair, chunk_no, mark_as_chunk_in_db=True):
            chunk_start_date = chunk_no * self.ONE_CHUNK_SECONDS
            chunk_end_date = chunk_start_date + self.ONE_CHUNK_SECONDS

            query = 'SELECT * FROM {} WHERE date >= ? AND date < ? ORDER BY date ASC'.format(pair)
            self.__cursor.execute(query, (chunk_start_date, chunk_end_date,))

            for row in self.__cursor:
                candlestick = self.__make_candlestick_from_row(row=row)
                self.__history[pair][candlestick.date] = candlestick

            if mark_as_chunk_in_db:
                self.__chunks_in_db[pair].add(chunk_no)
            self.__chunks_in_memory[pair].add(chunk_no)

        def persist_chunks(self):
            for pair, chunks_in_memory in self.__chunks_in_memory.items():
                for i, chunk_no in enumerate(sorted(chunks_in_memory)):
                    if chunk_no not in self.__chunks_in_db[pair]:
                        if i == 0:
                            start_from = self.get_pair_oldest_date(pair=pair) \
                                         % self.ONE_CHUNK_SECONDS
                            self.__persist_chunk(pair=pair,
                                                 chunk_no=chunk_no,
                                                 start_offset=start_from)
                        else:
                            self.__persist_chunk(pair=pair, chunk_no=chunk_no)

        def __persist_chunk(self, pair, chunk_no, start_offset=0):
            rows_to_insert = []
            start_date = chunk_no * self.ONE_CHUNK_SECONDS + start_offset
            for i in range(start_date, start_date + self.ONE_CHUNK_SECONDS, 300):
                try:
                    candlestick = self.__history[pair][i]
                    rows_to_insert.append(self.__make_row_from_candlestick(candlestick))
                except KeyError:
                    break

            self.__cursor.executemany(
                'INSERT OR REPLACE INTO {} VALUES (?, ?, ?, ?, ?, ?, ?, ?)'.format(pair),
                rows_to_insert
            )
            self.__conn.commit()

        @staticmethod
        def __make_candlestick_from_row(row):
            return Candlestick(date=row[0], high=row[1], low=row[2],
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
        def __get_db_connection():
            return sqlite3.connect(DATABASE_PATH)

        def __create_pair_table_if_not_exists(self, pair):
            self.__cursor.execute('''CREATE TABLE IF NOT EXISTS {}
                                    (date INTEGER PRIMARY KEY ASC, high REAL, low REAL, 
                                    open REAL, close REAL, base_volume REAL,
                                    quote_volume REAL, weighted_average REAL)'''.format(pair))
            self.__conn.commit()

    __instance = None

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = cls.__CandlestickStore()
        return cls.__instance

