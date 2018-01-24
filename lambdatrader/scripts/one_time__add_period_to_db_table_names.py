import sqlite3

DB_PATH = '/data/lambdatrader/sqlite/history.db'

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

table_names_query = "SELECT name FROM sqlite_master WHERE type='table'"
table_names = [row[0] for row in cursor.execute(table_names_query).fetchall()
               if row[0].find('BTC') >= 0]

for table_name in table_names:
    cursor.execute("ALTER TABLE '{}' RENAME TO '{}:M5'".format(table_name, table_name))
