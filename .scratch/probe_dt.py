import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from core.db import get_db_connection

cur = get_db_connection().cursor()
cur.execute("SHOW TABLES LIKE 'formtable_main_385%'")
for r in cur.fetchall():
    print('TABLE:', list(r.values())[0])
cur.execute("SHOW COLUMNS FROM formtable_main_385_dt1")
for r in cur.fetchall():
    print('  col:', r['Field'], r['Type'])
