import os
import sqlite3

from collections import defaultdict

from blist import sorteddict

from models.candlestick import Candlestick
from utils import get_project_directory, date_floor, date_ceil

DATABASE_PATH = os.path.join(get_project_directory(), 'db', 'history.db')


class CandlestickStore:

    ONE_CHUNK_SECONDS = 86400

    def __init__(self):
        self.conn = self.__get_db_connection()
        self.cursor = self.conn.cursor()

        self.history = defaultdict(sorteddict)
        self.chunks_in_memory = defaultdict(set)
        self.chunks_in_db = defaultdict(set)
        self.synced_pairs = set()

    def append_candlestick(self, pair, candlestick: Candlestick):
        self.__sync_pair_if_not_synced(pair)
        oldest_date = self.get_pair_oldest_date(pair)

        if oldest_date is not None and oldest_date != candlestick.date - 300:
            error_message = 'Candlestick date {} is not the increment of latest date {}'
            raise ValueError(error_message.format(candlestick.date, oldest_date))

        self.history[pair][candlestick.date] = candlestick
        self.chunks_in_memory[pair].add(self.__get_chunk_no(candlestick.date))

    def get_candlestick(self, pair, date):
        self.__sync_pair_if_not_synced(pair)
        if self.__get_chunk_no(date) not in self.chunks_in_memory[pair]:
            self.__load_chunk(pair, self.__get_chunk_no(date))
        return self.history[pair][date]

    def get_pair_oldest_date(self, pair):
        self.__sync_pair_if_not_synced(pair)
        pair_history = self.history[pair]

        if len(pair_history.keys()) == 0:
            return None

        return pair_history[pair_history.keys()[0]].date

    def get_pair_newest_date(self, pair):
        self.__sync_pair_if_not_synced(pair)
        pair_history = self.history[pair]

        if len(pair_history.keys()) == 0:
            return None

        return pair_history[pair_history.keys()[-1]].date

    @staticmethod
    def __get_chunk_no(date):
        return date % 86400

    def __sync_pair_if_not_synced(self, pair):
        if pair not in self.synced_pairs:
            self.__sync_pair(pair)

    def __sync_pair(self, pair):
        self.__create_pair_table_if_not_exists(pair)

        if self.__pair_table_is_empty(pair):
            return

        self.__load_pair_first_chunk(pair)
        self.__load_pair_last_chunk(pair)

    def __load_pair_first_chunk(self, pair):
        self.__load_chunk(pair, self.__get_chunk_no(self.__get_pair_oldest_date_from_db(pair)))

    def __load_pair_last_chunk(self, pair):
        self.__load_chunk(pair,
                          self.__get_chunk_no(self.__get_pair_newest_date_from_db(pair)),
                          mark_as_chunk_in_db=False)

    def __pair_table_is_empty(self, pair):
        query = 'SELECT * FROM ? LIMIT 1'
        self.cursor.execute(query, (pair,))
        return self.cursor.rowcount == 0

    def __get_pair_oldest_date_from_db(self, pair):
        query = 'SELECT date FROM ? ORDER BY date ASC LIMIT 1'
        return self.cursor.execute(query, (pair,)).fetchone()[0]

    def __get_pair_newest_date_from_db(self, pair):
        query = 'SELECT date FROM ? ORDER BY date DESC LIMIT 1'
        return self.cursor.execute(query, (pair,)).fetchone()[0]

    def __load_chunk(self, pair, chunk_no, mark_as_chunk_in_db=True):
        chunk_start_date = chunk_no * self.ONE_CHUNK_SECONDS
        chunk_end_date = chunk_start_date + self.ONE_CHUNK_SECONDS

        query = 'SELECT * FROM ? WHERE date >= ? AND date < ? ORDER BY date ASC'
        self.cursor.execute(query, (pair, chunk_start_date, chunk_end_date,))

        for row in self.cursor:
            candlestick = self.__make_candlestick_from_row(row)
            self.history[pair][candlestick.date] = candlestick

        if mark_as_chunk_in_db:
            self.chunks_in_db[pair].add(chunk_no)
        self.chunks_in_memory[pair].add(chunk_no)

    def __persist_chunks(self):
        for pair, chunks_in_memory in self.chunks_in_memory.items():
            for chunk_no in chunks_in_memory:
                if chunk_no not in self.chunks_in_db[pair]:
                    self.__persist_chunk(pair, chunk_no)

    def __persist_chunk(self, pair, chunk_no):
        insert_arguments = []
        start_date = chunk_no * self.ONE_CHUNK_SECONDS
        for i in range(start_date, start_date + self.ONE_CHUNK_SECONDS, 300):
            try:
                candlestick = self.history[pair][i]
                insert_arguments.append((pair,) + self.__make_row_from_candlestick(candlestick))
            except KeyError:
                break

        self.cursor.executemany('INSERT INTO ? VALUES (?, ?, ?, ?, ?, ?, ?, ?)', insert_arguments)
        self.conn.commit()

    @staticmethod
    def __make_candlestick_from_row(row):
        return Candlestick(date=row[0], high=row[1], low=row[2],
                           _open=row[3], close=row[4], base_volume=row[5],
                           quote_volume=row[6], weighted_average=row[7])

    @staticmethod
    def __make_row_from_candlestick(candlestick):
        return (candlestick.date, candlestick.high, candlestick.low, candlestick._open,
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
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS ?
                                (date INTEGER PRIMARY KEY ASC, high REAL, low REAL, 
                                open REAL, close REAL, base_volume REAL,
                                quote_volume REAL, weighted_average REAL)''', (pair,))
        self.conn.commit()
