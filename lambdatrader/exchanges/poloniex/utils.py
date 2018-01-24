from queue import Queue, Empty
from threading import Thread

from lambdatrader.account import UnableToFillImmediately
from lambdatrader.utilities.exceptions import (
    ConnectionTimeout, RequestLimitExceeded, InvalidJSONResponse, InternalError,
)


class APICallExecutor:
    class __APICallExecutor:
        def __init__(self):
            self.queued_calls = Queue()
            t = Thread(target=self.__executor)
            t.start()

        def call(self, call):
            result_queue = Queue()
            self.__register(call=call, return_queue=result_queue)
            result = result_queue.get()
            if result[1]:
                raise result[1]
            else:
                return result[0]

        def __register(self, call, return_queue):
            self.queued_calls.put((call, return_queue))

        def __executor(self):
            while True:
                try:
                    _function, return_queue = self.queued_calls.get(timeout=0.1)
                    try:
                        return_queue.put((_function(), None))
                    except Exception as e:
                        return_queue.put((None, e))
                except Empty:
                    pass

    __instance = None

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = cls.__APICallExecutor()
        return cls.__instance


def map_exception(e):
    e_str = str(e)
    if 'Unable to fill order completely.' in e_str:
        return UnableToFillImmediately(e_str)
    elif 'Connection timed out.' in e_str:
        return ConnectionTimeout(e_str)
    elif 'Please do not make more than' in e_str:
        return RequestLimitExceeded(e_str)
    elif 'Invalid json response' in e_str:
        return InvalidJSONResponse(e_str)
    elif 'Internal error. Please try again.' in e_str:
        return InternalError(e_str)
    else:
        return e
