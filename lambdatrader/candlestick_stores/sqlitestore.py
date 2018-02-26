import functools
import os
import sqlite3

import pandas as pd

from lambdatrader.config import CANDLESTICK_DB_DIRECTORY
from lambdatrader.constants import PeriodEnum, M5, M15, H, H4, D
from lambdatrader.exchanges.enums import ExchangeEnum
from lambdatrader.models.candlestick import Candlestick

DATABASE_DIR = CANDLESTICK_DB_DIRECTORY

if not os.path.isdir(DATABASE_DIR):
    os.makedirs(DATABASE_DIR, exist_ok=True)


class SQLiteCandlestickStore:

    class __SQLiteCandlestickStore:

        def __init__(self, exchange: ExchangeEnum):
            self._conn = self._get_db_connection(exchange=exchange)
            self._cursor = self._conn.cursor()

            self._existing_pair_periods_cache = set()

        def get_pairs(self):
            return set(map(self.pair_from_pair_period, self._get_pair_period_table_names()))

        def get_periods(self):
            return set(map(self.period_from_pair_period, self._get_pair_period_table_names()))

        def append_candlestick(self, pair, candlestick: Candlestick):
            period = candlestick.period
            pair_period = self.pair_period_name(pair, candlestick.period)

            self._create_pair_period_table_if_not_exists(pair_period)

            newest_date = self.get_pair_period_newest_date(pair=pair, period=period)

            if newest_date is not None and candlestick.date > newest_date + period.seconds():
                error_message = 'Candlestick date {} is not an increment of latest date {}'
                raise ValueError(error_message.format(candlestick.date, newest_date))

            self._add_candlesticks(pair_period, candlesticks=[candlestick])

        def _add_candlesticks(self, pair_period, candlesticks):
            rows_to_insert = map(self._make_row_from_candlestick, candlesticks)

            self._cursor.executemany(
                "INSERT OR REPLACE INTO '{}' VALUES (?, ?, ?, ?, ?, ?, ?, ?)".format(pair_period),
                rows_to_insert)
            self._conn.commit()

        def get_candlestick(self, pair, date, period=M5):
            pair_period = self.pair_period_name(pair, period)

            self._cursor.execute("SELECT * FROM '{}' WHERE date == ?"
                                 .format(pair_period), (date,))

            row = self._cursor.fetchone()
            if row is None:
                raise KeyError('{}:{}'.format(pair_period, date))

            candlestick = self._make_candlestick_from_row(period=period, row=row)
            return candlestick

        def get_candlestick_range(self, pair, start_date, end_date, period=M5):
            pair_period = self.pair_period_name(pair, period)

            self._cursor.execute("SELECT * FROM '{}' "
                                 "WHERE date >= ? AND date <= ? ORDER BY date ASC"
                                 .format(pair_period), (start_date, end_date))

            make_candlestick_func = functools.partial(self._make_candlestick_from_row, period)
            candlesticks = list(map(make_candlestick_func, self._cursor))
            if len(candlesticks) != int((end_date - start_date) / period.seconds()) + 1:
                raise KeyError('{}:{}-{}'.format(pair_period, start_date, end_date))
            return candlesticks

        def get_df(self, pair, start_date=None, end_date=None, period=M5, error_on_missing=False):
            query = self._get_df_sql_query(pair, start_date, end_date, period)
            df = pd.read_sql_query(query, index_col='date', parse_dates=['date'], con=self._conn)
            if error_on_missing:
                df_start = int(df.index[0].timestamp())
                df_end = int(df.index[-1].timestamp())
                if df_start != start_date or df_end != end_date:
                    raise KeyError('Missing Data')
            return df

        def get_agg_period_df(self, pair,
                              start_date=None, end_date=None, period=M5, error_on_missing=False):
            return self.get_agg_period_dfs(pair,
                                           start_date=start_date,
                                           end_date=end_date,
                                           periods=[period],
                                           error_on_missing=error_on_missing)[period]

        def get_agg_period_dfs(self, pair,
                               start_date=None, end_date=None, periods=(M5, M15, H, H4, D),
                               error_on_missing=False):
            m5_df = self.get_df(pair, start_date=start_date,
                                end_date=end_date, period=M5, error_on_missing=error_on_missing)
            period_df_dict = {}
            for period in periods:
                if period is M5:
                    period_df_dict[M5] = m5_df
                else:
                    period_df_dict[period] = self._agg_period_df(m5_df, period)
            return period_df_dict

        @staticmethod
        def _agg_period_df(m5_df, period):
            pd_offset_mapping = {
                M15: '15T',
                H: 'H',
                H4: '4H',
                D: 'D',
            }
            ohlc_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'base_volume': 'sum',
                'quote_volume': 'sum',
                'weighted_average': 'mean',
            }
            cols = m5_df.columns.values.tolist()
            return (m5_df.resample(pd_offset_mapping[period], closed='right', label='right')
                    .agg(ohlc_dict)[cols])

        def _get_df_sql_query(self, pair, start_date, end_date, period):
            pair_period = self.pair_period_name(pair, period)

            if start_date is not None and end_date is not None:
                return ("SELECT * FROM '{}' WHERE date >= {} AND date <= {} ORDER BY date ASC"
                        .format(pair_period, start_date, end_date))
            elif start_date is None and end_date is not None:
                return ("SELECT * FROM '{}' WHERE date <= {} ORDER BY date ASC"
                        .format(pair_period, end_date))
            elif start_date is not None and end_date is None:
                return ("SELECT * FROM '{}' WHERE date >= {} ORDER BY date ASC"
                        .format(pair_period, start_date))
            else:
                return "SELECT * FROM '{}' ORDER BY date ASC".format(pair_period)

        def get_pair_period_oldest_date(self, pair, period=M5):
            pair_period = self.pair_period_name(pair, period)
            return self._get_pair_period_oldest_date_from_db(pair_period)

        def get_pair_period_newest_date(self, pair, period=M5):
            pair_period = self.pair_period_name(pair, period)
            return self._get_pair_period_newest_date_from_db(pair_period)

        def _get_pair_period_table_names(self):
            query = "SELECT name FROM sqlite_master WHERE type='table'"
            return [
                row[0] for row in self._cursor.execute(query).fetchall()
                if row[0].find('BTC') == 0
            ]

        def _get_pair_period_oldest_date_from_db(self, pair_period):
            query = "SELECT date FROM '{}' ORDER BY date ASC LIMIT 1".format(pair_period)
            fetched = self._cursor.execute(query).fetchone()
            return None if fetched is None else fetched[0]

        def _get_pair_period_newest_date_from_db(self, pair_period):
            query = "SELECT date FROM '{}' ORDER BY date DESC LIMIT 1".format(pair_period)
            fetched = self._cursor.execute(query).fetchone()
            return None if fetched is None else fetched[0]

        @staticmethod
        def _make_candlestick_from_row(period, row):
            return Candlestick(period=period, date=row[0], high=row[1], low=row[2],
                               _open=row[3], close=row[4], base_volume=row[5],
                               quote_volume=row[6], weighted_average=row[7])

        @staticmethod
        def _make_row_from_candlestick(candlestick):
            return (candlestick.date, candlestick.high, candlestick.low, candlestick.open,
                    candlestick.close, candlestick.base_volume, candlestick.quote_volume,
                    candlestick.weighted_average)

        @staticmethod
        def _get_db_connection(exchange: ExchangeEnum):
            db_path = os.path.join(DATABASE_DIR, '{}.db'.format(exchange.name))
            return sqlite3.connect(db_path)

        def _create_pair_period_table_if_not_exists(self, pair_period):
            if pair_period not in self._existing_pair_periods_cache:
                self._cursor.execute('''CREATE TABLE IF NOT EXISTS '{}'
                                        (date INTEGER PRIMARY KEY ASC, high REAL, low REAL, 
                                        open REAL, close REAL, base_volume REAL,
                                        quote_volume REAL, weighted_average REAL)'''
                                     .format(pair_period))
                self._conn.commit()
                self._existing_pair_periods_cache.add(pair_period)

        @staticmethod
        def pair_period_name(pair, period: PeriodEnum):
            return '{}:{}'.format(pair, period.name)

        @staticmethod
        def pair_from_pair_period(pair_period):
            return pair_period[:pair_period.index(':')]

        @staticmethod
        def period_from_pair_period(pair_period):
            return PeriodEnum.from_name(pair_period[pair_period.index(':')+1:])

    _instances = {}

    @classmethod
    def get_for_exchange(cls, exchange: ExchangeEnum=ExchangeEnum.POLONIEX):
        if exchange not in cls._instances:
            cls._instances[exchange] = cls.__SQLiteCandlestickStore(exchange=exchange)
        return cls._instances[exchange]
