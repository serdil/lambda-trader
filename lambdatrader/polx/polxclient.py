from poloniex import Poloniex

from lambdatrader.config import POLONIEX_API_SECRET, POLONIEX_API_KEY

polo = Poloniex(key=POLONIEX_API_KEY, secret=POLONIEX_API_SECRET)
