from enum import Enum


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
    VALUE_TAKE_PROFIT = 3


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


class FailureExitType(Enum):
    COMBINED_AND = 1
    COMBINED_OR = 2
    TIMEOUT_STOP_LOSS = 3


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
    def __init__(self, timeout):
        super().__init__(_type=FailureExitType.TIMEOUT_STOP_LOSS)
        self.timeout = timeout


class TradeSignal:
    def __init__(self, exchange, pair, entry: Entry, success_exit: SuccessExit,
                 failure_exit: FailureExit, good_for=FIVE_MINUTES):
        self.exchange = exchange
        self.pair = pair
        self.entry = entry
        self.success_exit = success_exit
        self.failure_exit = failure_exit
        self.good_for = good_for
