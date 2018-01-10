from functools import lru_cache

from pymongo import MongoClient

from lambdatrader.config import MONGODB_URI


@lru_cache(maxsize=1)
def get_mongo_client():
    return MongoClient(MONGODB_URI)


@lru_cache(maxsize=1)
def get_default_db():
    return get_mongo_client().default
