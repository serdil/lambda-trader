import pprint

import logging

from lambdatrader.backtesting.account import BacktestingAccount
from lambdatrader.backtesting.marketinfo import BacktestingMarketInfo
from lambdatrader.config import BACKTESTING_NUM_DAYS
from lambdatrader.evaluation.utils import statistics_over_periods, period_statistics
from lambdatrader.executors.executors import SignalExecutor
from lambdatrader.history.store import CandlestickStore
from lambdatrader.backtesting import backtest
from lambdatrader.signals.signals import RetracementSignalGenerator

logger = logging.getLogger(__name__)

ONE_DAY = 24 * 3600

BACKTEST_NUM_DAYS = ONE_DAY * BACKTESTING_NUM_DAYS

market_info = BacktestingMarketInfo(candlestick_store=CandlestickStore.get_instance())

account = BacktestingAccount(market_info=market_info, balances={'BTC': 100})

start_date = market_info.get_max_pair_end_time() - 1 * BACKTEST_NUM_DAYS
end_date = market_info.get_max_pair_end_time() - 0 * BACKTEST_NUM_DAYS

signal_generators = [
        RetracementSignalGenerator(market_info=market_info)
    ]
signal_executor = SignalExecutor(market_info=market_info, account=account)

backtest.backtest(account=account, market_info=market_info, signal_generators=signal_generators,
                  signal_executor=signal_executor, start=start_date, end=end_date)


logger.info('Backtest Complete!')

logger.info('Estimated Balance: %f', account.get_estimated_balance())
logger.info('Open Orders:%s', str(list(account.get_open_orders())))

trading_info = signal_executor.get_trading_info()


logger.info(str(trading_info))

stats = period_statistics(trading_info=trading_info)


logger.info('Statistics over whole trading period:')
logger.info(pprint.pformat(stats))

stats_over_weekly_periods = statistics_over_periods(trading_info=trading_info, period_days=7)

logger.info('Statistics over weekly periods:')
logger.info(pprint.pformat(stats_over_weekly_periods))

stats_over_monthly_periods = statistics_over_periods(trading_info=trading_info, period_days=30)

logger.info('Statistics over monthly periods:')
logger.info(pprint.pformat(stats_over_monthly_periods))
