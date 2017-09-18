from queue import Queue, Empty
from threading import Thread


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