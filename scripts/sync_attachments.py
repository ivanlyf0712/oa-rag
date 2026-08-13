#!/usr/bin/env python3
"""Sync files under uploads/contracts/<ref_no>/<field_name>/ into the
contract_attachments MySQL table so the app can resolve attachments by ref_no.

Directory layout expected:
    uploads/contracts/<ref_no>/<field_name>/<file_name>

Files directly under <ref_no>/ (no field-name subdir) are imported with
field_name='unspecified'. Windows ':Zone.Identifier' sidecar files are skipped.

The script is idempotent: rows whose (ref_no, field_name, file_path) already
exist in contract_attachments are skipped.

Usage:
    venv/bin/python scripts/sync_attachments.py [--dry-run]
"""
import mimetypes
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(ROOT, '.env'))

from core.db import get_db_connection  # noqa: E402

UPLOAD_ROOT = os.path.join(ROOT, 'uploads', 'contracts')
FORM_TABLES = ('formtable_main_332', 'formtable_main_385')  # contract main forms
SKIP_SUFFIX = ':Zone.Identifier'
MISSING_FIELD_DIR = 'unspecified'


def find_contract_id(cur, ref_no):
    """Resolve the OA contract id from the main form tables by RefNo."""
    for table in FORM_TABLES:
        cur.execute('SELECT id FROM %s WHERE RefNo = %%s' % table, (ref_no,))
        row = cur.fetchone()
        if row:
            return list(row.values())[0] if isinstance(row, dict) else row[0]
    return None


def collect_files():
    """Walk the upload tree -> list of dicts(ref_no, field_name, file_name, file_path)."""
    entries = []
    for dirpath, _dirnames, filenames in os.walk(UPLOAD_ROOT):
        for name in filenames:
            if name.endswith(SKIP_SUFFIX) or name.startswith('.'):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, UPLOAD_ROOT)
            parts = rel.split(os.sep)
            if len(parts) < 2:
                print('SKIP (not under a ref_no dir): %s' % rel)
                continue
            ref_no = parts[0]
            field_name = parts[1] if len(parts) >= 3 else MISSING_FIELD_DIR
            entries.append({
                'ref_no': ref_no,
                'field_name': field_name,
                'file_name': name,
                'file_path': full,
            })
    return entries


def main():
    dry_run = '--dry-run' in sys.argv
    entries = collect_files()
    print('Found %d file(s) under %s' % (len(entries), UPLOAD_ROOT))

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute('SELECT ref_no, field_name, file_path FROM contract_attachments')
    existing = set()
    for r in cur.fetchall():
        vals = list(r.values()) if isinstance(r, dict) else list(r)
        existing.add(tuple(vals))

    inserted = skipped = 0
    for e in sorted(entries, key=lambda x: (x['ref_no'], x['field_name'], x['file_name'])):
        key = (e['ref_no'], e['field_name'], e['file_path'])
        if key in existing:
            skipped += 1
            continue
        contract_id = find_contract_id(cur, e['ref_no'])
        if contract_id is None:
            print('SKIP (ref_no %s not found in %s): %s'
                  % (e['ref_no'], '/'.join(FORM_TABLES), e['file_name']))
            skipped += 1
            continue
        mime, _enc = mimetypes.guess_type(e['file_name'])
        size = os.path.getsize(e['file_path'])
        if e['field_name'] == MISSING_FIELD_DIR:
            print('WARN (no field-name subdir, using "%s"): %s/%s'
                  % (MISSING_FIELD_DIR, e['ref_no'], e['file_name']))
        if dry_run:
            print('DRY  insert: %s | %s | %s (%d bytes)'
                  % (e['ref_no'], e['field_name'], e['file_name'], size))
            inserted += 1
            continue
        cur.execute(
            'INSERT INTO contract_attachments'
            ' (contract_id, ref_no, field_name, file_name, file_path, mime_type, file_size)'
            ' VALUES (%s, %s, %s, %s, %s, %s, %s)',
            (contract_id, e['ref_no'], e['field_name'], e['file_name'],
             e['file_path'], mime, size),
        )
        inserted += 1
        print('OK   insert: %s | %s | %s' % (e['ref_no'], e['field_name'], e['file_name']))

    if not dry_run:
        conn.commit()
    conn.close()
    print('Done: %d inserted, %d skipped%s.' % (inserted, skipped, ' (dry-run)' if dry_run else ''))


if __name__ == '__main__':
    main()
