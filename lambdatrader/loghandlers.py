import logging.handlers
import os

from telegram_handler import TelegramHandler

from lambdatrader.config import (
    TELEGRAM_TOKEN, BOT_NAME, TELEGRAM_CHAT_IDS
)
from lambdatrader.utils import get_project_directory

_1MB = 1024 * 1024
_5MB = 5 * _1MB
_256MB = 256 * _1MB
_1GB = 1024 * _1MB

LOG_FOLDER_PATH = os.path.join(get_project_directory(), 'log')
DEBUG_LOG_PATH = os.path.join(LOG_FOLDER_PATH, 'debug.log')
INFO_LOG_PATH = os.path.join(LOG_FOLDER_PATH, 'info.log')

formatter = logging.Formatter(
    BOT_NAME + ': ' + '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

file_handler_debug = logging.handlers.RotatingFileHandler(
    DEBUG_LOG_PATH, maxBytes=_256MB, backupCount=1
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

telegram_handlers = []

for chat_id in TELEGRAM_CHAT_IDS:
    handler = TelegramHandler(token=TELEGRAM_TOKEN, chat_id=chat_id)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    telegram_handlers.append(handler)


def add_all_handlers(logger):
    logger.addHandler(file_handler_debug)
    logger.addHandler(file_handler_info)
    logger.addHandler(console_handler)
    for handler in telegram_handlers:
        logger.addHandler(handler)


def get_logger_with_all_handlers(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not len(logger.handlers):
        add_all_handlers(logger)
    return logger
