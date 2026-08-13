import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from core.db import get_db_connection

FIELDS = ['reviewtier', 'FinanceApprovalLevel', 'MatrixFinanceApprovalLevel',
          'SignoffLevel', 'DisplayLevel', 'BusinessApprovalLevel']

cur = get_db_connection().cursor()
for f in FIELDS:
    print('==== ' + f + ' ====')
    cur.execute('SELECT ' + f + ' AS v, COUNT(*) AS n FROM formtable_main_385 GROUP BY ' + f + ' ORDER BY ' + f)
    dist = [(r['v'], r['n']) for r in cur.fetchall()]
    print('  dist:', ', '.join(str(v) + ' x' + str(n) for v, n in dist))
    for v, n in dist:
        if v is None:
            continue
        cur.execute('SELECT RefNo, TitleReferenceNoOfContract t, ContractAmountHKD a, status s FROM formtable_main_385 WHERE ' + f + ' = %s ORDER BY RefNo LIMIT 2', (v,))
        for r in cur.fetchall():
            print('   ' + str(v) + ': ' + str(r['RefNo']) + ' | ' + str(r['t'])[:48] + ' | HKD ' + str(r['a']) + ' | st ' + str(r['s']))
