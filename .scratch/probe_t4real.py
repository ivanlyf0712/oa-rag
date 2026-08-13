import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from core.db import get_db_connection

FIELDS = 'RefNo, TitleReferenceNoOfContract, CounterpartyName, businessunit, ContractAmountHKD, status'
cur = get_db_connection().cursor()
for t in (4, 3):
    sql = 'SELECT ' + FIELDS + ' FROM formtable_main_385 WHERE contract_type = ' + str(t) + " AND TitleReferenceNoOfContract NOT LIKE '%est%' ORDER BY RefNo DESC LIMIT 6"
    cur.execute(sql)
    print('=== contract_type = ' + str(t) + ' (non-test) ===')
    for r in cur.fetchall():
        print(r['RefNo'], '|', str(r['TitleReferenceNoOfContract'])[:62], '|', str(r['CounterpartyName'])[:32], '| BU:', r['businessunit'], '| HKD', r['ContractAmountHKD'], '| st', r['status'])
