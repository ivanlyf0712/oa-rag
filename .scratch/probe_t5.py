import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from core.db import get_db_connection

FIELDS = 'RefNo, TitleReferenceNoOfContract, CounterpartyName, businessunit, ContractAmountHKD, contractstartdate, contractenddate, status, ProductServices'
cur = get_db_connection().cursor()
for t in (5, 4, 7):
    cur.execute('SELECT ' + FIELDS + ' FROM formtable_main_385 WHERE contract_type = ' + str(t) + ' ORDER BY RefNo LIMIT 8')
    print('=== contract_type = ' + str(t) + ' ===')
    for r in cur.fetchall():
        title = str(r['TitleReferenceNoOfContract'])[:60]
        party = str(r['CounterpartyName'])[:35]
        prod = str(r['ProductServices'])[:45]
        print(r['RefNo'], '|', title, '|', party, '| BU:', r['businessunit'], '| HKD', r['ContractAmountHKD'], '|', r['contractstartdate'], '->', r['contractenddate'], '| st', r['status'])
        print('    prod:', prod)
