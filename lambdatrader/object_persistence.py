import pickle

from lambdatrader.loghandlers import get_logger_with_all_handlers
from lambdatrader.mongo_client import get_default_db


logger = get_logger_with_all_handlers(__name__)

_module_initialized = False
db = None
binary_objects = None


def _module_init():
    global _module_initialized, db, binary_objects
    if not _module_initialized:
        db = get_default_db()
        binary_objects = db.binary_objects
        binary_objects.create_index('key', unique=True)
        _module_initialized = True


def create_binary_object_document(key, object):
    _module_init()
    return {
        'key': key,
        'object': pickle.dumps(object, protocol=4)
    }


def get_object_with_key(key):
    _module_init()
    logger.debug('get_object_with_key:%s', key)
    result = binary_objects.find_one({'key': key})

    if result == None:
        return None

    logger.debug('got_object_from_db:%s', key)

    saved_object = pickle.loads(result['object'])
    return saved_object


def save_object_with_key(key, object):
    _module_init()
    logger.debug('save_object_with_key:%s', key)
    new_doc = create_binary_object_document(key, object)
    binary_objects.find_one_and_update({'key': key}, {'$set': {'object': new_doc['object']}},
                                       upsert=True)
