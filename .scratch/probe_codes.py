import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from core.db import get_db_connection

COLS = ['contracttype', 'contract_type', 'BusinessApprovalLevel', 'FinanceApprovalLevel',
        'DetailFinanceApprovalLevel', 'MatrixFinanceApprovalLevel', 'BusinessPreApprovalLevel',
        'SignoffLevel', 'DisplayLevel', 'reviewtier', 'RiskEndorser',
        'BusinessSecurityLevel', 'BusinessApproverSecurityLevel']

cur = get_db_connection().cursor()
for col in COLS:
    sql = 'SELECT ' + col + ' AS v, COUNT(*) AS n, MIN(RefNo) AS ex FROM formtable_main_385 GROUP BY ' + col + ' ORDER BY n DESC LIMIT 14'
    try:
        cur.execute(sql)
        rows = cur.fetchall()
    except Exception as e:
        print(col + ': ERR ' + str(e))
        continue
    vals = ', '.join(str(r['v']) + ' x' + str(r['n']) + ' (e.g. ' + str(r['ex']) + ')' for r in rows)
    print(col + ': ' + vals)
