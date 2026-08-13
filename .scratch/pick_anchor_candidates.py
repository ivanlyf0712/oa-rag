import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from core.db import get_db_connection

cur = get_db_connection().cursor()

# 1. Find real (non-test) contracts for each contract_type 3,4,5,7
print('=== contract_type 3,4,5,7 (non-test, real titles) ===')
for t in (3, 4, 5, 7):
    cur.execute("SELECT RefNo t, TitleReferenceNoOfContract tt, CounterpartyName cp, ContractAmountHKD a FROM formtable_main_385 WHERE contract_type = %s AND (TitleReferenceNoOfContract NOT LIKE '%%est%%' OR TitleReferenceNoOfContract IS NULL) AND TitleReferenceNoOfContract IS NOT NULL AND TitleReferenceNoOfContract != '' ORDER BY a DESC LIMIT 3", (t,))
    for r in cur.fetchall():
        print(f'  t={t}: {r["t"]} | {str(r["tt"])[:60]} | {str(r["cp"])[:30]} | HKD {r["a"]}')

# 2. Find all real-looking contracts (non-test, good title) for the ambiguous codes
print()
print('=== best real examples for each remaining field ===')

for field, label in [('BusinessApprovalLevel', 'BizApprLvl'), ('FinanceApprovalLevel', 'FinApprLvl'),
                     ('reviewtier', 'ReviewTier'), ('SignoffLevel', 'Signoff'), ('DisplayLevel', 'Display'), ('MatrixFinanceApprovalLevel', 'MatrixFin')]:
    cur.execute(f"""
        SELECT RefNo, TitleReferenceNoOfContract t, CounterpartyName cp, ContractAmountHKD a,
               contract_type ct, BusinessApprovalLevel bal, FinanceApprovalLevel fal,
               reviewtier rt, SignoffLevel sl, DisplayLevel dl, MatrixFinanceApprovalLevel mfl
        FROM formtable_main_385
        WHERE TitleReferenceNoOfContract NOT LIKE '%%est%%'
          AND TitleReferenceNoOfContract IS NOT NULL
          AND TitleReferenceNoOfContract != ''
          AND {field} IS NOT NULL
        ORDER BY a DESC
        LIMIT 20
    """)
    seen = set()
    print(f'\n--- {label} (no test title, sorted by amount DESC) ---')
    for r in cur.fetchall():
        key = r[field]
        if key not in seen:
            seen.add(key)
            print(f'  {key}: {r["RefNo"]} | {str(r["t"])[:55]:55s} | {str(r["cp"])[:25]:25s} | HKD {r["a"]:>20,.2f}')
        if len(seen) >= 8:
            break
