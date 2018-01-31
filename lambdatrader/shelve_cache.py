import os
import shelve

from lambdatrader.config import SHELVE_CACHE_DIRECTORY

CACHE_DIR = SHELVE_CACHE_DIRECTORY
SHELVE_PATH = os.path.join(CACHE_DIR, 'shelve_cache.dbm')

if not os.path.isdir(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)

shelve_dict = shelve.open(SHELVE_PATH, flag='c', protocol=None, writeback=False)


def shelve_cache_save(cache_key, obj):
    shelve_dict[repr(cache_key)] = obj


def shelve_cache_get(cache_key):
    return shelve_dict[repr(cache_key)]
