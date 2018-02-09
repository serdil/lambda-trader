from enum import Enum


class ExchangeEnum(Enum):
    BACKTESTING = 1
    POLONIEX = 2

BACKTESTING = ExchangeEnum.BACKTESTING
POLONIEX = ExchangeEnum.POLONIEX
