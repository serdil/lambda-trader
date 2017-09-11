from poloniex import Poloniex, Coach

from lambdatrader.config import POLONIEX_API_SECRET, POLONIEX_API_KEY

coach = Coach(timeFrame=1, callLimit=3)

polo = Poloniex(key=POLONIEX_API_KEY, secret=POLONIEX_API_SECRET, coach=coach)
