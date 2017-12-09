from enum import Enum
from uuid import uuid1

FIVE_MINUTES = 5 * 60


class EntryType:
    PRICE = 1


class Entry:
    def __init__(self, _type=EntryType.PRICE):
        self.type = _type


class PriceEntry(Entry):
    def __init__(self, price):
        super().__init__(_type=EntryType.PRICE)
        self.price = price


class SuccessExitType(Enum):
    COMBINED_AND = 1
    COMBINED_OR = 2
    PRICE_TAKE_PROFIT = 3


class SuccessExit:
    def __init__(self, _type: SuccessExitType):
        self.type = _type


class CombinedAndSuccessExit(SuccessExit):
    def __init__(self, success_exits):
        super().__init__(_type=SuccessExitType.COMBINED_AND)
        self.success_exits = success_exits[:]


class CombinedOrSuccessExit(SuccessExit):
    def __init__(self, success_exits):
        super().__init__(_type=SuccessExitType.COMBINED_OR)
        self.success_exits = success_exits[:]


class PriceTakeProfitSuccessExit(SuccessExit):
    def __init__(self, price):
        super().__init__(_type=SuccessExitType.PRICE_TAKE_PROFIT)
        self.price = price


class FailureExitType(Enum):
    COMBINED_AND = 1
    COMBINED_OR = 2
    TIMEOUT_STOP_LOSS = 3
    PRICE_STOP_LOSS = 4


class FailureExit:
    def __init__(self, _type: FailureExitType):
        self.type = _type


class CombinedAndFailureExit(FailureExit):
    def __init__(self, failure_exits):
        super().__init__(_type=FailureExitType.COMBINED_AND)
        self.failure_exits = failure_exits[:]


class CombinedOrFailureExit(FailureExit):
    def __init__(self, failure_exits):
        super().__init__(_type=FailureExitType.COMBINED_OR)
        self.failure_exits = failure_exits[:]


class TimeoutStopLossFailureExit(FailureExit):
    def __init__(self, timeout, sl_timeout):
        super().__init__(_type=FailureExitType.TIMEOUT_STOP_LOSS)
        self.timeout = timeout
        self.sl_timeout = sl_timeout


class PriceStopLossFailureExit(FailureExit):
    def __init__(self, price):
        super().__init__(_type=FailureExitType.PRICE_STOP_LOSS)
        self.price = price


class TradeSignal:
    def __init__(self, date, exchange, pair, entry: Entry, success_exit: SuccessExit,
                 failure_exit: FailureExit, good_for=FIVE_MINUTES, _id=None):
        self.date = date
        self.exchange = exchange
        self.pair = pair
        self.entry = entry
        self.success_exit = success_exit
        self.failure_exit = failure_exit
        self.good_for = good_for

        if _id:
            self.id = _id
        else:
            self.id = uuid1()

    def __repr__(self):
        return 'TradeSignal(date={},pair={})'.format(self.date, self.pair)
