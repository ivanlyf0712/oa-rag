import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from core.db import get_db_connection

cur = get_db_connection().cursor()
for c in ['BUApprovalGrade', 'BUApprovalGradeL1', 'BusinessApproverSecurityL1', 'BusinessApproverSecurityLevel']:
    try:
        cur.execute('SELECT ' + c + ' AS v, COUNT(*) AS n FROM formtable_main_385 GROUP BY ' + c + ' ORDER BY n DESC LIMIT 15')
        print(c + ': ' + ', '.join(str(r['v']) + ' x' + str(r['n']) for r in cur.fetchall()))
    except Exception as e:
        print(c + ': ERR ' + str(e))
print()
print('--- cross-tab BusinessApprovalLevel (code) vs BUApprovalGrade (text) ---')
cur.execute('SELECT BusinessApprovalLevel c, BUApprovalGrade g, COUNT(*) n FROM formtable_main_385 GROUP BY BusinessApprovalLevel, BUApprovalGrade ORDER BY c, n DESC')
for r in cur.fetchall():
    print('  code', str(r['c']), '<->', str(r['g']), 'x' + str(r['n']))
