from evaluation.statistics import Statistics
from models.tradinginfo import TradingInfo

ONE_DAY = 24 * 3600


def period_statistics(trading_info: TradingInfo, start=None, end=None):
    if start is None:
        start = trading_info.history_start
    if end is None:
        end = trading_info.history_end

    return Statistics(trading_info).calc_stats_for_period(start_date=start, end_date=end)


def statistics_over_periods(trading_info: TradingInfo, start=None, end=None, period_days=7):
    if start is None:
        start = trading_info.history_start
    if end is None:
        end = trading_info.history_end

    statistics = Statistics(trading_info)

    periods = []
    period_stats = []

    for date in range(start, end, period_days * ONE_DAY):
        periods.append((date, date + 7 * ONE_DAY))

    for period in periods:
        period_stats.append(statistics.calc_stats_for_period(period[0], period[1]))

    return statistics.calc_stats_over_periods(period_stats)
