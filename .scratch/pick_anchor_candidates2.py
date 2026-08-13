import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from core.db import get_db_connection

cur = get_db_connection().cursor()

# Best real-looking contracts for each code value to look up in the OA UI
fields_codes = {
    'BusinessApprovalLevel': [0, 1, 2, 4, 5, 7, 8, 9, 19],
    'FinanceApprovalLevel': [0, 3],
    'reviewtier': [2, 3, 4],
    'SignoffLevel': [2, 3, 4],
    'DisplayLevel': [2],
}

for field, codes in fields_codes.items():
    print(f'=== {field} ===')
    for code in codes:
        cur.execute(f"""
            SELECT RefNo, TitleReferenceNoOfContract t, CounterpartyName cp, ContractAmountHKD a,
                   contract_type ct
            FROM formtable_main_385
            WHERE {field} = %s
              AND (TitleReferenceNoOfContract NOT LIKE '%%est%%' OR TitleReferenceNoOfContract IS NULL)
              AND TitleReferenceNoOfContract IS NOT NULL
              AND TitleReferenceNoOfContract != ''
            ORDER BY a DESC
            LIMIT 1
        """, (code,))
        r = cur.fetchone()
        if r:
            print(f'  {code}: {r["RefNo"]} | {str(r["t"])[:60]} | {str(r["cp"])[:25]} | HKD {r["a"]} | ct={r["ct"]}')
        else:
            # fallback to test data
            cur.execute(f"SELECT RefNo, TitleReferenceNoOfContract t, CounterpartyName cp, ContractAmountHKD a, contract_type ct FROM formtable_main_385 WHERE {field} = %s ORDER BY a DESC LIMIT 1", (code,))
            r = cur.fetchone()
            if r:
                print(f'  {code}: {r["RefNo"]} | {str(r["t"])[:60]} | {str(r["cp"])[:25]} | HKD {r["a"]} | ct={r["ct"]} [TEST]')
            else:
                print(f'  {code}: no rows')
