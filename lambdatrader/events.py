from collections import defaultdict
from enum import Enum
from queue import Queue

from blist import sorteddict

from lambdatrader.config import BOT_IDENTIFIER


class EventManager:

    def __init__(self):
        self._events_dict = defaultdict(sorteddict)
        self._subscribers = defaultdict(set)
        self._event_queue = Queue()

    def event_loop(self):
        while True:
            event = self.event_loop().get(timeout=0.1)
            self._emit_event(event)

    def emit_event(self, event):
        self._event_queue.put(event)

    def _emit_event(self, event):
        self._events_dict[event.type][event.date] = event

        for subscriber in self._subscribers[event.type]:
            subscriber.receive_event(event)

        for subscriber in self._subscribers[EventType.ALL]:
            subscriber.receive_event(event)

    def subscribe_for_events(self, subscriber, event_types=None):
        if event_types is None:
            event_types = [EventType.ALL]

        for event_type in event_types:
            self._subscribers[event_type].add(subscriber)

    def persist_events(self):
        pass

    def get_events(self, event_types=None, events_since=0):
        pass


class EventType(Enum):
    ALL = 1
    TP_HIT = 2
    SL_HIT = 3
    TRADE_CLOSED = 4
    NEW_SIGNAL = 5


class BaseEvent:
    def __init__(self, date):
        self.source = BOT_IDENTIFIER
        self.date = date


class BaseSignalEvent(BaseEvent):
    def __init__(self, date, signal):
        super().__init__(date)
        self.signal = signal


class TPHitEvent(BaseSignalEvent):
    def __init__(self, date, signal):
        super().__init__(date, signal)
        self.signal = signal


class SLHitEvent(BaseSignalEvent):
    def __init__(self, date, signal):
        super().__init__(date, signal)


class TradeClosedEvent(BaseSignalEvent):
    def __init__(self, date, signal):
        super().__init__(date, signal)
