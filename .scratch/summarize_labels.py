# From the user's form data, extract the actual codes and their grade labels

# Anchor: CCA20250047 (BusinessApprovalLevel=4, test data said bal=4
#   Component 1: Business Approval Level = GM2
#   Component 2: Business Approval Level = GM1
#   DOA Required Business Approver Level = GM1
#   Finance: Entity Finance Head, BU Group Finance Head

# Anchor: CCA20260159 (BusinessApprovalLevel=5)
#   Component 1: Business Approval Level = GM2
#   DOA Required Business Approver Level = GM2
#   Contract Clearance Business Approver = M2
#   Finance Approval Level = Entity Finance Head

# BUT: CCA20250047 had BusinessApprovalLevel=4 with GM2/GM1 -> this is the SAME grade as CCA20260159 which has bal=5 with GM2
# That means the code does NOT map 1:1 to grade - it's more complex
# Let me check the actual DB values

from dotenv import load_dotenv
load_dotenv('/home/ivanleeyf/oa-rag/.env')
import sys, os
sys.path.insert(0, '/home/ivanleeyf/oa-rag')
from core.db import get_db_connection

cur = get_db_connection().cursor()

# Check BusinessApprovalLevel cross-tab with BUApprovalGradeL1
print('=== BusinessApprovalLevel vs BUApprovalGradeL1 (all rows) ===')
cur.execute('''
    SELECT BusinessApprovalLevel, BUApprovalGradeL1, COUNT(*) n
    FROM formtable_main_385
    WHERE BusinessApprovalLevel IS NOT NULL
    GROUP BY BusinessApprovalLevel, BUApprovalGradeL1
    ORDER BY BusinessApprovalLevel, n DESC
''')
for r in cur.fetchall():
    print(f'  bal={r["BusinessApprovalLevel"]:>2s}  grade={str(r["BUApprovalGradeL1"]):>4s}  x{r["n"]}')

print()
print('=== FinanceApprovalLevel vs FinanceLevel text columns ===')
# Check what columns might contain the finance level text
cur.execute("SHOW COLUMNS FROM formtable_main_385 LIKE '%Approval%'")
columns = [r['Field'] for r in cur.fetchall()]
for c in columns:
    cur.execute(f'SELECT {c} AS v, COUNT(*) n FROM formtable_main_385 WHERE {c} IS NOT NULL AND {c} != "" AND {c} NOT IN ("None","0","2","3","1") AND BusinessApprovalLevel IS NULL GROUP BY {c} ORDER BY n DESC LIMIT 5')
    rows = cur.fetchall()
    if rows:
        print(f'  {c}: ' + ', '.join(f'{r["v"]}x{r["n"]}' for r in rows))

print()
print('=== Specific contracts checked ===')
for ref in ['CCA20250046', 'CCA20250057', 'CCA20250047', 'CCA20260020', 'CCA20260159', 'CCA20260163']:
    cur.execute("SELECT RefNo, BusinessApprovalLevel, FinanceApprovalLevel, reviewtier, SignoffLevel, DisplayLevel, MatrixFinanceApprovalLevel, BUApprovalGradeL1, BUApprovalGrade FROM formtable_main_385 WHERE RefNo = %s", (ref,))
    r = cur.fetchone()
    if r:
        print(f'  {r["RefNo"]}: bal={r["BusinessApprovalLevel"]} fal={r["FinanceApprovalLevel"]} rt={r["reviewtier"]} sl={r["SignoffLevel"]} dl={r["DisplayLevel"]} mfl={r["MatrixFinanceApprovalLevel"]} gradeL1={r["BUApprovalGradeL1"]!r} grade={r["BUApprovalGrade"]!r}')
