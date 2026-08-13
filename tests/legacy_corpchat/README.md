# Legacy CorpChat tests (SUPERSEDED)

These tests target the removed  package (CorpChat message/chat
domain). They pre-date the oa-rag contract-domain port and no longer collect
().

They are kept here for reference only and are excluded from the default test
run. The contract-domain agent behavior is covered by:

- tests/test_contract_router.py  — search gate + 5-intent model + filter mapping
- tests/test_contract_agent.py   — CrossTableAgent manual-ReAct tool routing
- tests/test_agent_regression.py — public-seam regression gate + LLM-down degradation
- tests/test_risk_search.py      — risk_search pipeline
- tests/test_oa_app.py           — Streamlit app seams

Do not re-enable these without porting them to the contract domain.
