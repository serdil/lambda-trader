
def every_n_market_seconds(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            self = args[0]
            if self.market_date - wrapper._last_called_date > n:
                func_result = func(*args, *kwargs)
                wrapper._last_called_date = self.market_date
                return func_result
            else:
                return None
        wrapper._last_called_date = 0
        return wrapper
    return decorator
