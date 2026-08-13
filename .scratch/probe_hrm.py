import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from core.db import get_db_connection

cur = get_db_connection().cursor()
cur.execute("SHOW TABLES LIKE 'hrmresource'")
print('hrmresource table:', cur.fetchone())
if True:
    try:
        cur.execute('SELECT id, lastname FROM hrmresource WHERE id IN (25958,25952,25953,25957,23787,11478,11247,9806,22598)')
        for r in cur.fetchall():
            print(r['id'], '->', r['lastname'])
    except Exception as e:
        print('ERR', e)
