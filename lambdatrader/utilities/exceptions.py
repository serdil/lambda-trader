class TradingException(Exception):
    pass


class APIConnectionException(TradingException):
    pass


class RequestLimitExceeded(APIConnectionException):
    pass


class InvalidJSONResponse(APIConnectionException):
    pass


class ConnectionTimeout(APIConnectionException):
    pass


class InternalError(APIConnectionException):
    pass
