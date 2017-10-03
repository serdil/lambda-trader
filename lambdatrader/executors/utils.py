from lambdatrader.utilities.exceptions import (
    ConnectionTimeout, RequestLimitExceeded, InvalidJSONResponse,
)


def retry_on_exception(task, logger, exceptions=None):
    if exceptions is None:
        exceptions = [ConnectionTimeout, RequestLimitExceeded, InvalidJSONResponse]

    try:
        return task()
    except Exception as e:
        if type(e) in exceptions:
            logger.warning(str(e))
            return retry_on_exception(task=task, logger=logger, exceptions=exceptions)
        else:
            raise e
