import os, sys, statistics
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from core.db import get_db_connection

FIELDS = ['reviewtier', 'FinanceApprovalLevel', 'MatrixFinanceApprovalLevel',
          'SignoffLevel', 'DisplayLevel', 'BusinessApprovalLevel']

def parse_amt(s):
    if s is None:
        return None
    try:
        return float(str(s).replace(',', ''))
    except ValueError:
        return None

cur = get_db_connection().cursor()
cur.execute('SELECT ContractAmountHKD a, reviewtier, FinanceApprovalLevel, MatrixFinanceApprovalLevel, SignoffLevel, DisplayLevel, BusinessApprovalLevel FROM formtable_main_385')
rows = cur.fetchall()
for f in FIELDS:
    print('==== ' + f + ' vs amount (n, min / median / max HKD) ====')
    buckets = {}
    for r in rows:
        v = r[f]
        if v is None:
            continue
        a = parse_amt(r['a'])
        buckets.setdefault(v, [])
        if a is not None:
            buckets[v].append(a)
    for v in sorted(buckets):
        amts = buckets[v]
        if amts:
            print('   ' + str(v) + ': n=' + str(len(amts)) + '  ' + format(int(min(amts)), ',d') + ' / ' + format(int(statistics.median(amts)), ',d') + ' / ' + format(int(max(amts)), ',d'))
        else:
            print('   ' + str(v) + ': n=0')
