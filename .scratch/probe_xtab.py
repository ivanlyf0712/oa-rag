import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from core.db import get_db_connection

cur = get_db_connection().cursor()
print('--- BusinessApprovalLevel code vs BUApprovalGradeL1 text ---')
cur.execute('SELECT BusinessApprovalLevel c, BUApprovalGradeL1 g, COUNT(*) n FROM formtable_main_385 GROUP BY BusinessApprovalLevel, BUApprovalGradeL1 ORDER BY c, n DESC')
curmap = {}
for r in cur.fetchall():
    if r['c'] is None:
        continue
    curmap.setdefault(r['c'], []).append(str(r['g']) + ' x' + str(r['n']))
for c in sorted(curmap):
    print('  code ' + str(c) + ': ' + ', '.join(curmap[c]))
print()
print('--- FinanceApprovalLevel code vs BUFinanceApprover/GroupFinanceApprover filled? ---')
cur.execute('SELECT FinanceApprovalLevel c, COUNT(*) n, SUM(BUFinanceApprover IS NOT NULL AND BUFinanceApprover != "") bu, SUM(GroupFinanceApprover IS NOT NULL AND GroupFinanceApprover != "") grp FROM formtable_main_385 GROUP BY FinanceApprovalLevel ORDER BY c')
for r in cur.fetchall():
    print('  code ' + str(r['c']) + ': n=' + str(r['n']) + ' BU_fin_filled=' + str(r['bu']) + ' Group_fin_filled=' + str(r['grp']))
