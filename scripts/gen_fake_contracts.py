#!/usr/bin/env python3
"""Generate FAKE OA contract data for local development / demos.

The real data (MySQL dump, CSV exports, uploaded files, search index) is
never committed to git. This script recreates a deterministic, obviously
fictional dataset that exercises every app feature:

  * risk flags (Over5M / Over100M / IsRisksAccepted / FlagNeedLegal ...)
    using the exact coded-field names from core/db.py BOOLEAN_CODE_FIELDS
  * workflow statuses 1-7, dates (expired / ending soon / no end date)
  * amounts across the HK00k / 5M / 10M / 100M thresholds
  * narrative text fields used by core.db._extract_contract_text

Outputs (pick any combination):
  --sql      data/fake_seed.sql      CREATE TABLE + INSERTs for formtable_main_385
  --csv      data/fake_formtable_main_385.csv
  --uploads  uploads/contracts/<ref>/<field>/...  (minimal valid PDF/DOCX)

Default (no flags) = --sql --csv --uploads. Deterministic via --seed.

Usage:
  venv/bin/python scripts/gen_fake_contracts.py [--count 25] [--seed 42]
  mysql -u oa -p oa_rag < data/fake_seed.sql   # or: make db-import equivalent
  venv/bin/python scripts/build_index.py --force
"""
import argparse
import csv
import os
import random
import zipfile
from datetime import date, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLE = os.getenv("CONTRACTS_TABLE", "formtable_main_385")

# Columns written to SQL/CSV. Mirrors the columns core/db.py actually reads
# (normalization + decoded flags + narrative text). The real table has ~155
# columns; extras stay NULL and are irrelevant to the app.
CORE_COLUMNS = [
    "id", "requestId", "RefNo", "TitleReferenceNoOfContract",
    "CounterpartyName", "ProductServices", "businessunit", "Department",
    "ContractAmountHKD", "contractstartdate", "contractenddate",
    "requested_date", "requested_time", "Status", "contracttype",
    "isDeleteProcess", "reviewtier",
    "modedatacreater", "modedatacreatedate", "modedatacreatetime",
]

# Keep in sync with core/db.py BOOLEAN_CODE_FIELDS (0=no, 1=yes, 2=na).
CODE_COLUMNS = [
    "Over5M", "Over100M", "WithEndDate", "Saved",
    "IncludingExternalGuarantees", "IsAuthoritySufficient",
    "IsRisksAccepted", "IsRenew", "iscontractfinancial",
    "needapreliminaryreviewbygroupl", "PreliminaryReviewFlag",
    "preliminaryreviewflag2", "FlagNeedLegal", "FlagNeedGFN",
    "IfRelatedToData", "relatedtocapexpropertyleasingc",
    "generalpurchaseandoverhk50k", "unlimitedliabilitiesorliabilit",
    "allrelevantdocumentationhasbee", "ihaveread10points",
]

TEXT_COLUMNS = [
    "description", "KeyChanges", "assessmentprompt1", "assessmentprompt7",
    "ReasonToSubmitLT5M", "ReasonsNoEndDa", "NoteOnContractAmount",
]

COLUMNS = CORE_COLUMNS + CODE_COLUMNS + TEXT_COLUMNS

DEPARTMENTS = ["IT", "Finance", "Procurement", "HR", "Legal", "Marketing"]
CONTRACT_TYPES = ["General Purchase", "Service Agreement", "Licensing",
                  "Property Leasing", "Financial", "NDA"]
# Obviously fictional counterparties.
COUNTERPARTIES = [
    "Acme Fictitious Trading Co", "ExampleSoft Ltd", "Placeholder Logistics HK",
    "Demo Foods International", "Mockup Media Group", "Sample Systems Asia",
    "Testbench Engineering", "Fable Pharmaceuticals", "Lorem Consulting",
    "Ipsum Cloud Services",
]
PRODUCTS = [
    "Office laptop refresh (200 units)", "Cloud ERP subscription (3 years)",
    "Warehouse logistics outsourcing", "Catering services 2026-2028",
    "Media buying campaign", "HR payroll SaaS licence",
    "Network security audit", "Cold-chain distribution",
    "Data annotation services", "Print advertising package",
]


def _money(rng):
    """Amounts spread across the HK00k / 5M / 10M / 100M thresholds."""
    return rng.choice([
        rng.randint(80_000, 450_000),          # small
        rng.randint(500_000, 4_900_000),       # under 5M
        rng.randint(5_100_000, 9_900_000),     # over 5M
        rng.randint(10_000_000, 99_000_000),   # over 10M
        rng.randint(100_000_000, 150_000_000), # over 100M
    ])


def make_rows(count, seed):
    rng = random.Random(seed)
    rows = []
    for i in range(1, count + 1):
        amount = _money(rng)
        start = date(2024, 1, 1) + timedelta(days=rng.randint(0, 700))
        has_end = rng.random() > 0.15
        end = (start + timedelta(days=rng.randint(90, 1500))) if has_end else None
        # ~1 in 6 has risk accepted = no (the key risky-contract signal).
        risks_rejected = (i % 6 == 0)
        row = {
            "id": i,
            "requestId": 10_000 + i,
            "RefNo": "CCA%04d" % (2026 * 10_000 + i),
            "TitleReferenceNoOfContract": "Fake contract %03d" % i,
            "CounterpartyName": rng.choice(COUNTERPARTIES),
            "ProductServices": rng.choice(PRODUCTS),
            "businessunit": rng.choice(DEPARTMENTS),
            "Department": rng.randint(1, 8),
            "ContractAmountHKD": amount,
            "contractstartdate": start.isoformat(),
            "contractenddate": end.isoformat() if end else "",
            "requested_date": (start - timedelta(days=14)).isoformat(),
            "requested_time": "10:00:00",
            "Status": rng.randint(0, 7),
            "contracttype": rng.choice(CONTRACT_TYPES),
            "isDeleteProcess": 0,
            "reviewtier": rng.choice([1, 2, 3]),
            "modedatacreater": 1,
            "modedatacreatedate": (start - timedelta(days=20)).isoformat(),
            "modedatacreatetime": "09:30:00",
            # coded flags (0/1/2)
            "Over5M": 1 if amount > 5_000_000 else 0,
            "Over100M": 1 if amount > 100_000_000 else 0,
            "WithEndDate": 1 if has_end else 0,
            "Saved": 1,
            "IncludingExternalGuarantees": rng.choice([0, 0, 0, 1]),
            "IsAuthoritySufficient": 0 if rng.random() < 0.1 else 1,
            "IsRisksAccepted": 0 if risks_rejected else 1,
            "IsRenew": rng.choice([0, 1]),
            "iscontractfinancial": rng.choice([0, 0, 1]),
            "needapreliminaryreviewbygroupl": rng.choice([0, 1]),
            "PreliminaryReviewFlag": rng.choice([0, 1]),
            "preliminaryreviewflag2": rng.choice([0, 1]),
            "FlagNeedLegal": 1 if rng.random() < 0.25 else 0,
            "FlagNeedGFN": 1 if rng.random() < 0.15 else 0,
            "IfRelatedToData": rng.choice([0, 0, 1]),
            "relatedtocapexpropertyleasingc": rng.choice([0, 0, 1]),
            "generalpurchaseandoverhk50k": 1 if amount > 50_000 else 0,
            "unlimitedliabilitiesorliabilit": 1 if rng.random() < 0.1 else 0,
            "allrelevantdocumentationhasbee": 0 if rng.random() < 0.1 else 1,
            "ihaveread10points": 1,
            # narrative
            "description": "Fake contract for %s. Auto-generated test data." % rng.choice(PRODUCTS),
            "KeyChanges": "None (fake data)" if rng.random() < 0.5 else "Renewal with updated pricing (fake)",
            "assessmentprompt1": "Key risks: none identified (synthetic record).",
            "assessmentprompt7": "Mitigations documented in fake assessment.",
            "ReasonToSubmitLT5M": "Below threshold (fake)" if amount < 5_000_000 else "",
            "ReasonsNoEndDa": "Evergreen service (fake)" if not has_end else "",
            "NoteOnContractAmount": "Estimated spend (fake)",
        }
        rows.append(row)
    return rows


def _sql_val(v):
    if v is None or v == "":
        return "NULL"
    if isinstance(v, (int, float)):
        return str(v)
    return "'" + str(v).replace("\\", "\\\\").replace("'", "''") + "'"


def write_sql(rows, path):
    cols = ", ".join(COLUMNS)
    col_defs = ",\n  ".join(
        "%s %s" % (c, "BIGINT" if c in ("id", "requestId") else
                   ("DOUBLE" if c == "ContractAmountHKD" else
                    ("INT" if c in CODE_COLUMNS else "TEXT")))
        for c in COLUMNS)
    lines = [
        "-- FAKE seed data generated by scripts/gen_fake_contracts.py",
        "CREATE TABLE IF NOT EXISTS %s (\n  %s\n) CHARSET=utf8mb4;" % (TABLE, col_defs),
        "DELETE FROM %s;" % TABLE,
    ]
    for r in rows:
        lines.append("INSERT INTO %s (%s) VALUES (%s);" % (
            TABLE, cols, ", ".join(_sql_val(r[c]) for c in COLUMNS)))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def _pdf_bytes(text):
    """Minimal single-page PDF with a correct xref table (pypdf-readable)."""
    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length %d>>stream\nBT /F1 14 Tf 72 720 Td (%s) Tj ET\nendstream"
        % (len(text) + 33, text.encode("latin-1", "replace")),
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += ("%d 0 obj\n" % n).encode() + body + b"\nendobj\n"
    xref = len(out)
    out += ("xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)).encode()
    for off in offsets:
        out += ("%010d 00000 n \n" % off).encode()
    out += ("trailer<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF"
            % (len(objs) + 1, xref)).encode()
    return bytes(out)


def _docx_bytes(text):
    """Minimal valid .docx (OOXML zip)."""
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="xml" ContentType="application/xml"/>'
                   '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                   "</Types>")
        z.writestr("_rels/.rels",
                   '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                   "</Relationships>")
        z.writestr("word/document.xml",
                   '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                   "<w:body><w:p><w:r><w:t>" + text + "</w:t></w:r></w:p></w:body></w:document>")
    return buf.getvalue()


def write_uploads(rows, root):
    for r in rows:
        ref = r["RefNo"]
        label = "FAKE %s %s - test data only" % (ref, r["CounterpartyName"])
        for field, name, blob in (
            ("signedcontract", "%s_signed.pdf" % ref, _pdf_bytes(label)),
            ("DraftContract", "%s_draft.docx" % ref, _docx_bytes(label)),
        ):
            d = os.path.join(root, ref, field)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, name), "wb") as fh:
                fh.write(blob)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--count", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sql", action="store_true", help="write data/fake_seed.sql")
    ap.add_argument("--csv", action="store_true", help="write data/fake_formtable_main_385.csv")
    ap.add_argument("--uploads", action="store_true", help="write uploads/contracts/ dummy files")
    args = ap.parse_args()
    if not (args.sql or args.csv or args.uploads):
        args.sql = args.csv = args.uploads = True

    rows = make_rows(args.count, args.seed)
    if args.sql:
        out = os.path.join(ROOT, "data", "fake_seed.sql")
        write_sql(rows, out)
        print("wrote %s (%d rows)" % (out, len(rows)))
    if args.csv:
        out = os.path.join(ROOT, "data", "fake_formtable_main_385.csv")
        write_csv(rows, out)
        print("wrote %s" % out)
    if args.uploads:
        root = os.path.join(ROOT, "uploads", "contracts")
        write_uploads(rows, root)
        print("wrote dummy attachments under %s" % root)
    print("Next: mysql -u <user> -p <db> < data/fake_seed.sql && "
          "venv/bin/python scripts/build_index.py --force")


if __name__ == "__main__":
    main()
