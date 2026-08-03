#!/usr/bin/env python3
"""
Onyx Data Ingest Script - Upload contacts & messages tables as CSV files
to Onyx File Connectors, then verify the Agent can retrieve the data.

Usage:
    python3 ingest.py                     # upload & verify
    python3 ingest.py --no-verify         # upload only, skip verification
    python3 ingest.py --verify-only       # skip upload, just check search results

Prerequisites:
    - Onyx running at http://localhost:3000
    - API key in key.txt or set ONYX_API_KEY env var
    - Database (contacts & messages tables) accessible via the DB config
"""

import io
import csv
import os
import sys
import json
import time
import argparse
import traceback
from datetime import datetime

import requests

# ── Paths ────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(SCRIPT_DIR, "key.txt")

# ── Onyx Config ──────────────────────────────────────────────────
ONYX_URL = "http://localhost:3000"
API_BASE = f"{ONYX_URL}/api/manage/admin"

# Connector IDs (from Onyx Admin API: GET /api/manage/admin/connector)
CONTACTS_CONNECTOR_ID = 3   # FileConnector for contacts.csv
MESSAGES_CONNECTOR_ID = 4   # FileConnector for messages.csv

# ── Database Config (same as ocr/core/config.py) ─────────────────
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "ocr",
    "password": "ocrpass",
    "dbname": "invoices",
}

# ── Verification Search Queries ──────────────────────────────────
VERIFY_QUERIES = {
    "contacts": {
        "query": "傅健",
        "expected_in_result": "傅健",
        "description": "contact name 傅健",
    },
    "messages": {
        "query": "詐騙",  # "scam" in Chinese
        "expected_in_result": "詐騙",
        "description": "scam message content",
    },
}


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def get_api_key() -> str:
    """Read API key from key.txt or ONYX_API_KEY env var."""
    env_key = os.environ.get("ONYX_API_KEY")
    if env_key:
        return env_key
    try:
        with open(KEY_FILE, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"❌ API key file not found at {KEY_FILE}")
        print("   Create the file with your Onyx PAT, or set ONYX_API_KEY env var.")
        sys.exit(1)


def api_headers() -> dict:
    return {
        "Authorization": f"Bearer {get_api_key()}",
    }


def get_table_data(table_name: str):
    """Read table from PostgreSQL. Returns (column_names, rows)."""
    try:
        import psycopg2
    except ImportError:
        print("❌ psycopg2 not installed. Run: pip install psycopg2-binary")
        sys.exit(1)

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(f'SELECT * FROM "{table_name}"')
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()
    return columns, rows


def rows_to_csv_buffer(columns, rows) -> io.StringIO:
    """Write rows to an in-memory CSV buffer."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    writer.writerows(rows)
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════════════
#  Upload
# ═══════════════════════════════════════════════════════════════════

def upload_csv(connector_id: int, table_name: str, columns, rows) -> bool:
    """
    Upload a CSV to Onyx File Connector via the Admin API.
    The upload automatically triggers re-indexing.
    Returns True on success.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{table_name}_{timestamp}.csv"

    csv_buf = rows_to_csv_buffer(columns, rows)

    url = f"{API_BASE}/connector/{connector_id}/files/update"
    files = {
        "files": (filename, csv_buf.getvalue(), "text/csv"),
    }
    data = {
        "file_ids_to_remove": "[]",
    }

    print(f"  📤 Uploading {filename} ({len(rows)} rows, {len(columns)} cols)...")
    try:
        resp = requests.post(url, files=files, data=data, headers=api_headers(), timeout=60)
        if resp.status_code == 200:
            result = resp.json()
            file_names = result.get("file_names", [])
            print(f"     ✅ Uploaded successfully")
            print(f"     📄 Files on connector: {file_names}")
            return True
        else:
            print(f"     ❌ Upload failed (HTTP {resp.status_code}): {resp.text[:500]}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"     ❌ Network error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
#  Verify — check if Onyx search can find uploaded data
# ═══════════════════════════════════════════════════════════════════

def verify_agent_can_see_data(table_name: str) -> bool:
    """
    Poll the Onyx search API until the uploaded data is findable
    or we give up after several retries.
    """
    verify_info = VERIFY_QUERIES.get(table_name)
    if not verify_info:
        print(f"  ⚠️  No verification query configured for '{table_name}', skipping.")
        return True

    query = verify_info["query"]
    expected = verify_info["expected_in_result"]

    search_url = f"{ONYX_URL}/api/search"
    headers = api_headers()
    headers["Content-Type"] = "application/json"

    payload = {
        "query": query,
        "max_results": 10,
    }

    max_retries = 12       # ~2 minutes total waiting
    retry_delay = 10       # seconds between retries

    for attempt in range(1, max_retries + 1):
        print(f"  🔍 Searching Onyx for '{verify_info['description']}' "
              f"(attempt {attempt}/{max_retries})...")
        try:
            resp = requests.post(search_url, json=payload, headers=headers,
                                 timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                results = data if isinstance(data, list) else data.get("results", [])

                if results:
                    print(f"     Found {len(results)} result(s)!")
                    for i, r in enumerate(results[:2]):
                        title = r.get("title", r.get("name", ""))
                        snippet = (r.get("content", r.get("text", "")) or "")[:120]
                        print(f"       [{i+1}] {title}")
                        print(f"           {snippet}...")

                # Check if expected keyword appears
                found = False
                for r in results:
                    content = r.get("content", r.get("text", "")) or ""
                    title = r.get("title", r.get("name", "")) or ""
                    if expected in content or expected in title:
                        found = True
                        break

                if found:
                    print(f"     ✅ Found '{expected}' in search results!")
                    return True
                elif not results:
                    print(f"     ⏳ No results yet (indexing in progress)...")
                else:
                    print(f"     ⏳ '{expected}' not found yet (indexing in progress)...")
            elif resp.status_code == 503:
                print(f"     ⏳ Service temporarily unavailable (indexing)...")
            else:
                print(f"     ⚠️  Search API returned HTTP {resp.status_code}: "
                      f"{resp.text[:200]}")
        except requests.exceptions.Timeout:
            print(f"     ⏳ Search timed out (indexing may be in progress)...")
        except requests.exceptions.ConnectionError as e:
            print(f"     ⚠️  Connection error (retrying): {e}")
            time.sleep(5)
            continue
        except Exception as e:
            print(f"     ⚠️  Search error: {e}")

        if attempt < max_retries:
            print(f"     ⏳ Waiting {retry_delay}s before retry...")
            time.sleep(retry_delay)

    print(f"     ❌ Could not confirm data indexed after {max_retries} attempts.")
    print(f"     💡 Check indexing status at {ONYX_URL}/admin/connectors")
    return False


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Upload CSV data to Onyx and verify agent can see it")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip search verification after upload")
    parser.add_argument("--verify-only", action="store_true",
                        help="Skip upload; search for existing data only")
    args = parser.parse_args()

    print("=" * 55)
    print("  Onyx Data Ingest — CorpChat Pipeline")
    print("=" * 55)

    api_key = get_api_key()
    key_preview = api_key[:12] + "..." if len(api_key) > 15 else api_key
    print(f"\n🔑 API key: {key_preview}")
    print(f"🔄 Connectors: contacts(ID={CONTACTS_CONNECTOR_ID}), "
          f"messages(ID={MESSAGES_CONNECTOR_ID})")

    tables_to_process = [
        ("contacts", CONTACTS_CONNECTOR_ID),
        ("messages", MESSAGES_CONNECTOR_ID),
    ]

    results = {}

    for table_name, connector_id in tables_to_process:
        print(f"\n{'─' * 50}")
        print(f"📋 Table: {table_name} (connector {connector_id})")
        print(f"{'─' * 50}")

        # ── Upload ──
        if not args.verify_only:
            try:
                columns, rows = get_table_data(table_name)
                print(f"   📊 Read {len(rows)} rows, {len(columns)} columns")
            except Exception as e:
                print(f"   ❌ DB error: {e}\n{traceback.format_exc()}")
                results[table_name] = False
                continue

            if not rows:
                print(f"   ⚠️  Table empty, marking as passed.")
                results[table_name] = True
                continue

            upload_ok = upload_csv(connector_id, table_name, columns, rows)
            if not upload_ok:
                results[table_name] = False
                continue
        else:
            print("   ⏭️  Skipping upload (--verify-only)")

        # ── Verify ──
        if not args.no_verify:
            verify_ok = verify_agent_can_see_data(table_name)
            results[table_name] = verify_ok
        else:
            print("   ⏭️  Skipping verification (--no-verify)")
            results[table_name] = True

    # ── Summary ──
    print(f"\n{'=' * 55}")
    print("  Summary")
    print(f"{'=' * 55}")
    all_ok = True
    for table, ok in results.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {table}")
        if not ok:
            all_ok = False

    if all_ok and results:
        print(f"\n🎉 All tables processed successfully!")
        print(f"   Data uploaded to Onyx connectors and indexing triggered.")
        print(f"   Check the Onyx UI at {ONYX_URL}/admin/connectors for indexing status.")
        print(f"   Once indexed, the 'Enterprise Chat RAG' agent can answer questions.")
    elif results:
        print(f"\n⚠️  Some steps had issues. See logs above.")
        sys.exit(1)
    else:
        print(f"\n⚠️  No tables were processed.")
        sys.exit(1)


if __name__ == "__main__":
    main()