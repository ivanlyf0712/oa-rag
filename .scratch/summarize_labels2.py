from dotenv import load_dotenv
load_dotenv('/home/ivanleeyf/oa-rag/.env')
import sys, os
sys.path.insert(0, '/home/ivanleeyf/oa-rag')
from core.db import get_db_connection

cur = get_db_connection().cursor()

print('=== BusinessApprovalLevel vs BUApprovalGradeL1 ===')
cur.execute("SELECT BusinessApprovalLevel, BUApprovalGradeL1, COUNT(*) n FROM formtable_main_385 WHERE BusinessApprovalLevel IS NOT NULL GROUP BY BusinessApprovalLevel, BUApprovalGradeL1 ORDER BY BusinessApprovalLevel, n DESC")
for r in cur.fetchall():
    print('  bal=' + str(r['BusinessApprovalLevel']) + '  grade=' + str(r['BUApprovalGradeL1']) + '  x' + str(r['n']))

print()
print('=== Specific contracts ===')
for ref in ['CCA20250046', 'CCA20250057', 'CCA20250047', 'CCA20260159', 'CCA20260163', 'CCA20250020']:
    cur.execute("SELECT RefNo, BusinessApprovalLevel, FinanceApprovalLevel, reviewtier, SignoffLevel, DisplayLevel, MatrixFinanceApprovalLevel, BUApprovalGradeL1, BUApprovalGrade FROM formtable_main_385 WHERE RefNo = %s", (ref,))
    r = cur.fetchone()
    if r:
        print('  ' + str(r['RefNo']) + ': bal=' + str(r['BusinessApprovalLevel']) + ' fal=' + str(r['FinanceApprovalLevel']) + ' rt=' + str(r['reviewtier']) + ' sl=' + str(r['SignoffLevel']) + ' dl=' + str(r['DisplayLevel']) + ' mfl=' + str(r['MatrixFinanceApprovalLevel']) + ' gradeL1=' + repr(r['BUApprovalGradeL1']) + ' grade=' + repr(r['BUApprovalGrade']))
