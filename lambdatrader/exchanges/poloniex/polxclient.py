from poloniex import Poloniex, Coach

from lambdatrader.config import (
    POLONIEX_API_SECRET, POLONIEX_API_KEY, POLX_COACH_TIMEFRAME, POLX_COACH_CALL_LIMIT,
)

coach = Coach(timeFrame=POLX_COACH_TIMEFRAME, callLimit=POLX_COACH_CALL_LIMIT)

polo = Poloniex(key=POLONIEX_API_KEY, secret=POLONIEX_API_SECRET, coach=coach)
