from poloniex import Poloniex, Coach

from lambdatrader.config import POLONIEX_API_SECRET, POLONIEX_API_KEY

coach = Coach(timeFrame=2.0, callLimit=2)

polo = Poloniex(key=POLONIEX_API_KEY, secret=POLONIEX_API_SECRET, coach=coach)
