from pymongo import MongoClient

from lambdatrader.config import MONGODB_URI

mongo_client = MongoClient(MONGODB_URI)
default_db = mongo_client.default


def get_mongo_client():
    return mongo_client


def get_default_db():
    return default_db
