import logging.handlers

from telegram_handler import TelegramHandler

from config import TELEGRAM_CHAT_ID, TELEGRAM_TOKEN

_1MB = 1024 * 1024 * 1024
_5MB = 5 * _1MB
_1GB = 1024 * _1MB

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler_debug = logging.handlers.RotatingFileHandler('log/debug.log', maxBytes=_1GB, backupCount=2)
file_handler_debug.setLevel(logging.DEBUG)
file_handler_debug.setFormatter(formatter)

file_handler_info= logging.handlers.RotatingFileHandler('log/info.log', maxBytes=_5MB, backupCount=10)
file_handler_info.setLevel(logging.INFO)
file_handler_info.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

telegram_handler = TelegramHandler(token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)
telegram_handler.setLevel(logging.INFO)
telegram_handler.setFormatter(formatter)


def add_all_handlers(logger):
    logger.addHandler(file_handler_debug)
    logger.addHandler(file_handler_info)
    logger.addHandler(console_handler)
    logger.addHandler(telegram_handler)


def get_logger_with_all_handlers(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not len(logger.handlers):
        add_all_handlers(logger)
    return logger