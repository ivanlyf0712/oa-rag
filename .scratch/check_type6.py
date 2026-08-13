from dotenv import load_dotenv; load_dotenv('/home/ivanleeyf/oa-rag/.env')
import sys, os; sys.path.insert(0, '/home/ivanleeyf/oa-rag')
from core.db import get_db_connection
cur = get_db_connection().cursor()
cur.execute("SELECT contract_type, COUNT(*) n FROM formtable_main_385 GROUP BY contract_type ORDER BY contract_type")
for r in cur.fetchall():
    print(str(r['contract_type']) + ': ' + str(r['n']))
