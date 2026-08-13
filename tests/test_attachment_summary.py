"""Tests for apps.attachment_summary: attachment choice, text extraction
(PDF/DOCX), two-section summary, graceful fallback, and caching.

All DB access and LLM calls are faked; fixtures are real PDF/DOCX files
generated in tmp_path via fitz and python-docx. No network or live DB needed.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apps.attachment_summary import (
    attachment_label,
    choose_attachment,
    extract_text,
    human_file_size,
    resolve_attachment_path,
    summarize_contract_with_attachment,
)


# ─────────────────────────────────────────────────────────────────────
# Fixture file builders
# ─────────────────────────────────────────────────────────────────────
def _make_pdf(path: str, text: str = "This agreement is between Alpha Ltd and Beta Corp.") -> str:
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()
    return path


def _make_docx(path: str, text: str = "This services agreement is made between Gamma and Delta.") -> str:
    import docx
    d = docx.Document()
    d.add_paragraph(text)
    d.save(path)
    return path


class _FakeCursor:
    """A minimal cursor that returns pre-seeded rows for matching SQL."""

    def __init__(self, rows):
        self._rows = rows
        self._last = []
        self.description = [("field_name",), ("file_name",), ("file_path",), ("mime_type",), ("file_size",)]

    def execute(self, sql, params=()):
        if "contract_id = %s" in sql:
            key = params[0]
            self._last = [r for r in self._rows if r.get("contract_id") == key]
        elif "ref_no = %s" in sql:
            key = params[0]
            self._last = [r for r in self._rows if r.get("ref_no") == key]
        else:
            self._last = []

    def fetchall(self):
        cols = [d[0] for d in self.description]
        return [tuple(r.get(c) for c in cols) for r in self._last]


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _FakeCursor(self._rows)

    def close(self):
        pass


def _db(rows):
    """Return a db_connect callable yielding a fake connection seeded with rows."""
    return lambda: _FakeConn(rows)


class _ScriptedLLM:
    """Fake chat model returning canned content based on the prompt."""

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        user = messages[-1][1] if messages else ""

        class _Resp:
            pass

        r = _Resp()
        if "summarizing a signed contract document" in messages[0][1]:
            r.content = "This is a document summary of the signed contract."
        else:
            r.content = "This is a risk assessment based on the recorded tags."
        return r


# ─────────────────────────────────────────────────────────────────────
# Text extraction
# ─────────────────────────────────────────────────────────────────────
def test_extract_pdf(tmp_path):
    pdf = _make_pdf(str(tmp_path / "a.pdf"))
    out = extract_text(pdf, "application/pdf")
    assert out["error"] is None
    assert "Alpha Ltd" in out["text"]
    assert out["truncated"] is False


def test_extract_docx(tmp_path):
    docx = _make_docx(str(tmp_path / "b.docx"))
    out = extract_text(docx, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert out["error"] is None
    assert "Gamma" in out["text"]


def test_extract_missing_file():
    out = extract_text("/nonexistent/file.pdf", "application/pdf")
    assert out["error"] == "file not found"
    assert out["text"] == ""


def test_extract_cap_truncates(tmp_path):
    # build a multi-page PDF so extracted text reliably exceeds the cap
    import fitz
    pdf = str(tmp_path / "big.pdf")
    doc = fitz.open()
    for i in range(6):
        page = doc.new_page()
        page.insert_text((72, 72), ("Line %d of the contract body text. " % i) * 3)
    doc.save(pdf)
    doc.close()
    out = extract_text(pdf, "application/pdf", cap=200)
    assert out["truncated"] is True
    assert len(out["text"]) <= 200


# ─────────────────────────────────────────────────────────────────────
# Attachment choice (priority / existence / fallback)
# ─────────────────────────────────────────────────────────────────────
def test_choose_prefers_signed(tmp_path):
    signed = _make_pdf(str(tmp_path / "signed.pdf"), "signed")
    draft = _make_pdf(str(tmp_path / "draft.pdf"), "draft")
    rows = [
        {"contract_id": 1, "ref_no": "R1", "field_name": "signedcontract", "file_name": "s.pdf", "file_path": signed, "mime_type": "application/pdf", "file_size": 1},
        {"contract_id": 1, "ref_no": "R1", "field_name": "DraftContract", "file_name": "d.pdf", "file_path": draft, "mime_type": "application/pdf", "file_size": 1},
    ]
    att = choose_attachment(1, "R1", db_connect=_db(rows))
    assert att is not None
    assert att["field_name"] == "signedcontract"


def test_choose_skips_missing_signed_and_falls_to_draft(tmp_path):
    draft = _make_pdf(str(tmp_path / "draft.pdf"), "draft")
    rows = [
        {"contract_id": 2, "ref_no": "R2", "field_name": "signedcontract", "file_name": "s.pdf", "file_path": str(tmp_path / "gone.pdf"), "mime_type": "application/pdf", "file_size": 1},
        {"contract_id": 2, "ref_no": "R2", "field_name": "DraftContract", "file_name": "d.pdf", "file_path": draft, "mime_type": "application/pdf", "file_size": 1},
    ]
    att = choose_attachment(2, "R2", db_connect=_db(rows))
    assert att["field_name"] == "DraftContract"


def test_choose_ref_no_fallback_when_id_misses(tmp_path):
    f = _make_pdf(str(tmp_path / "x.pdf"), "x")
    rows = [
        {"contract_id": 999, "ref_no": "R3", "field_name": "signedcontract", "file_name": "x.pdf", "file_path": f, "mime_type": "application/pdf", "file_size": 1},
    ]
    # query by id=3 (no match), then ref_no R3 (match)
    att = choose_attachment(3, "R3", db_connect=_db(rows))
    assert att is not None and att["file_name"] == "x.pdf"


def test_choose_none_when_no_files_exist():
    rows = [
        {"contract_id": 4, "ref_no": "R4", "field_name": "signedcontract", "file_name": "s.pdf", "file_path": "/gone/s.pdf", "mime_type": "application/pdf", "file_size": 1},
    ]
    assert choose_attachment(4, "R4", db_connect=_db(rows)) is None


# ─────────────────────────────────────────────────────────────────────
# Two-section summary
# ─────────────────────────────────────────────────────────────────────
def _row(tmp_path, ref="R10", cid=10):
    pdf = _make_pdf(str(tmp_path / "att.pdf"), "This contract is with Omega Ltd for 100000 HKD.")
    return pd.Series({
        "id": cid,
        "ref_no": ref,
        "title": "Omega supply agreement",
        "counterparty_name": "Omega Ltd",
        "department": "IT",
        "amount_label": "HK00k",
        "risk_severity": "high",
        "risk_score": 42,
        "IsRisksAccepted": "no",
        "FlagNeedLegal": "yes",
        "matched_signals": [],
    }), pdf


def _attach_rows(cid, ref, path):
    return [{"contract_id": cid, "ref_no": ref, "field_name": "signedcontract",
             "file_name": os.path.basename(path), "file_path": path,
             "mime_type": "application/pdf", "file_size": 1}]


def test_summary_two_sections_with_llm(tmp_path, monkeypatch):
    row, pdf = _row(tmp_path)
    rows = _attach_rows(10, "R10", pdf)
    from apps import attachment_summary as mod
    llm = _ScriptedLLM()
    out = mod.summarize_contract_with_attachment(row, llm=llm, db_connect=_db(rows))
    assert out["document_summary"] is not None
    assert "risk assessment" in out["summary"].lower() or "Risk assessment" in out["summary"]
    assert "Document summary" in out["summary"]
    assert out["attachment"]["field_name"] == "signedcontract"
    assert out["notice"] is None


def test_summary_fallback_when_attachment_missing(tmp_path):
    row, _ = _row(tmp_path, ref="R11", cid=11)
    from apps import attachment_summary as mod
    out = mod.summarize_contract_with_attachment(row, llm=_ScriptedLLM(), db_connect=_db([]))
    assert out["document_summary"] is None
    assert out["notice"] is not None
    assert "risk tags only" in out["notice"]
    # still has a risk assessment section
    assert "Risk assessment" in out["summary"]


def test_summary_fallback_when_extraction_fails(tmp_path):
    row, _ = _row(tmp_path, ref="R12", cid=12)
    rows = _attach_rows(12, "R12", str(tmp_path / "gone.pdf"))
    # file_path points to nonexistent file → choose_attachment returns None (no existing file)
    from apps import attachment_summary as mod
    out = mod.summarize_contract_with_attachment(row, llm=_ScriptedLLM(), db_connect=_db(rows))
    assert out["notice"] is not None


def test_summary_caching(tmp_path):
    row, pdf = _row(tmp_path, ref="R13", cid=13)
    rows = _attach_rows(13, "R13", pdf)
    from apps import attachment_summary as mod
    llm = _ScriptedLLM()
    cache = {}
    out1 = mod.summarize_contract_with_attachment(row, llm=llm, db_connect=_db(rows), cache=cache)
    calls_after_first = llm.calls
    out2 = mod.summarize_contract_with_attachment(row, llm=llm, db_connect=_db(rows), cache=cache)
    # second call served from cache: no new LLM calls
    assert llm.calls == calls_after_first
    assert out1 is out2


def test_list_attachments_returns_all_rows_by_ref():
    """Public wrapper returns every attachment row for the contract."""
    from apps.attachment_summary import list_attachments
    rows = [
        {"ref_no": "R1", "field_name": "signedcontract", "file_name": "signed.pdf",
         "file_path": "/x/signed.pdf", "mime_type": "application/pdf", "file_size": 10},
        {"ref_no": "R1", "field_name": "DraftContract", "file_name": "draft.docx",
         "file_path": "/x/draft.docx", "mime_type": "application/msword", "file_size": 20},
        {"ref_no": "R2", "field_name": "signedcontract", "file_name": "other.pdf",
         "file_path": "/x/other.pdf", "mime_type": "application/pdf", "file_size": 5},
    ]
    atts = list_attachments(ref_no="R1", db_connect=_db(rows))
    assert [a["file_name"] for a in atts] == ["signed.pdf", "draft.docx"]



# ─────────────────────────────────────────────────────────────────────
# UPLOADS_ROOT path remap
# ─────────────────────────────────────────────────────────────────────
def test_resolve_path_as_is_when_exists(tmp_path):
    f = tmp_path / "x.pdf"
    f.write_bytes(b"%PDF")
    assert resolve_attachment_path(str(f)) == str(f)


def test_resolve_path_remaps_under_uploads_root(tmp_path, monkeypatch):
    real = tmp_path / "uploads" / "contracts" / "CCA1" / "signedcontract" / "s.pdf"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"%PDF")
    monkeypatch.setenv("UPLOADS_ROOT", str(tmp_path / "uploads"))
    stored = "/home/someone/oa-rag/uploads/contracts/CCA1/signedcontract/s.pdf"
    assert resolve_attachment_path(stored) == str(real)


def test_resolve_path_without_marker_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("UPLOADS_ROOT", str(tmp_path))
    assert resolve_attachment_path("/totally/absent/f.pdf") == "/totally/absent/f.pdf"


def test_choose_attachment_uses_remap(tmp_path, monkeypatch):
    real = tmp_path / "uploads" / "contracts" / "CCA1" / "signedcontract" / "s.pdf"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"%PDF")
    monkeypatch.setenv("UPLOADS_ROOT", str(tmp_path / "uploads"))
    rows = [{"contract_id": 7, "ref_no": "CCA1", "field_name": "signedcontract",
             "file_name": "s.pdf",
             "file_path": "/old/host/oa-rag/uploads/contracts/CCA1/signedcontract/s.pdf",
             "mime_type": "application/pdf", "file_size": 4}]
    chosen = choose_attachment(7, "CCA1", db_connect=_db(rows))
    assert chosen is not None
    assert chosen["file_path"] == str(real)


def test_choose_attachment_mixed_case_draft_matches_priority(tmp_path):
    draft = _make_docx(str(tmp_path / "d.docx"))
    rows = [{"contract_id": 8, "ref_no": "R8", "field_name": "DraftContract",
             "file_name": "d.docx", "file_path": draft,
             "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
             "file_size": 1}]
    chosen = choose_attachment(8, "R8", db_connect=_db(rows))
    assert chosen is not None
    assert chosen["file_name"] == "d.docx"


# ─────────────────────────────────────────────────────────────────────
# Attachment labels and sizes
# ─────────────────────────────────────────────────────────────────────
def test_attachment_label_mapping():
    assert attachment_label("signedcontract") == "Signed contract"
    assert attachment_label("finalversioncontract") == "Final version"
    assert attachment_label("DraftContract") == "Draft"
    assert attachment_label("unspecified") == "Other attachment"


def test_attachment_label_fallbacks():
    assert attachment_label("appendix_b") == "Appendix B"
    assert attachment_label("") == "Other attachment"
    assert attachment_label(None) == "Other attachment"


def test_human_file_size():
    assert human_file_size(None) == ""
    assert human_file_size(0) == ""
    assert human_file_size(512) == "512 B"
    assert human_file_size(2048) == "2.0 KB"
    assert human_file_size(3 * 1024 * 1024) == "3.0 MB"


# ─────────────────────────────────────────────────────────────────────
# Cache invalidation on attachment change
# ─────────────────────────────────────────────────────────────────────
def test_summary_cache_invalidates_on_mtime_change(tmp_path):
    row, pdf = _row(tmp_path, ref="R14", cid=14)
    rows = _attach_rows(14, "R14", pdf)
    from apps import attachment_summary as mod
    llm = _ScriptedLLM()
    cache = {}
    mod.summarize_contract_with_attachment(row, llm=llm, db_connect=_db(rows), cache=cache)
    calls_after_first = llm.calls
    mtime = os.path.getmtime(pdf)
    os.utime(pdf, (mtime + 5, mtime + 5))
    mod.summarize_contract_with_attachment(row, llm=llm, db_connect=_db(rows), cache=cache)
    assert llm.calls > calls_after_first
