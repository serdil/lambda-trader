import pickle

from loghandlers import get_logger_with_all_handlers
from mongo.mongo_client import get_default_db

logger = get_logger_with_all_handlers(__name__)

db = get_default_db()

binary_objects = db.binary_objects

binary_objects.create_index('key', unique=True)


def create_binary_object_document(key, object):
    return {
        'key': key,
        'object': pickle.dumps(object, protocol=4)
    }


def get_object_with_key(key):
    logger.debug('get_object_with_key:', key)
    result = binary_objects.find_one({'key': key})
    if result == None:
        return None
    logger.debug('got_object_from_db:', key)

    saved_object = pickle.loads(result['object'])
    return saved_object


def save_object_with_key(key, object):
    logger.debug('save_object_with_key:', key)
    binary_objects.replace_one({'key': key}, object, upsert=True)
