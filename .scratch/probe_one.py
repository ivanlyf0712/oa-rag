import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from core.db import get_db_connection

cur = get_db_connection().cursor()
cur.execute("SELECT RefNo, contract_type, BusinessApprovalLevel, FinanceApprovalLevel, reviewtier, SignoffLevel, DisplayLevel, MatrixFinanceApprovalLevel FROM formtable_main_385 WHERE RefNo = 'CCA20250005'")
print(cur.fetchone())
