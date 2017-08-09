import logging.handlers
import os

from telegram_handler import TelegramHandler

from lambdatrader.config import (
    TELEGRAM_TOKEN, BOT_NAME, TELEGRAM_CHAT_ID_1, TELEGRAM_CHAT_ID_2,
)
from lambdatrader.utils import get_project_directory

_1MB = 1024 * 1024 * 1024
_5MB = 5 * _1MB
_1GB = 1024 * _1MB

LOG_FOLDER_PATH = os.path.join(get_project_directory(), 'log')
DEBUG_LOG_PATH = os.path.join(LOG_FOLDER_PATH, 'debug.log')
INFO_LOG_PATH = os.path.join(LOG_FOLDER_PATH, 'info.log')

formatter = logging.Formatter(
    BOT_NAME + ': ' + '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

file_handler_debug = logging.handlers.RotatingFileHandler(
    DEBUG_LOG_PATH, maxBytes=_1GB, backupCount=1
)
file_handler_debug.setLevel(logging.DEBUG)
file_handler_debug.setFormatter(formatter)

file_handler_info = logging.handlers.RotatingFileHandler(
    INFO_LOG_PATH, maxBytes=_5MB, backupCount=10
)
file_handler_info.setLevel(logging.INFO)
file_handler_info.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

telegram_handler_1 = TelegramHandler(token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID_1)
telegram_handler_1.setLevel(logging.INFO)
telegram_handler_1.setFormatter(formatter)

telegram_handler_2 = TelegramHandler(token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID_2)
telegram_handler_2.setLevel(logging.INFO)
telegram_handler_2.setFormatter(formatter)


def add_all_handlers(logger):
    logger.addHandler(file_handler_debug)
    logger.addHandler(file_handler_info)
    logger.addHandler(console_handler)
    logger.addHandler(telegram_handler_1)
    logger.addHandler(telegram_handler_2)


def get_logger_with_all_handlers(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not len(logger.handlers):
        add_all_handlers(logger)
    return logger
