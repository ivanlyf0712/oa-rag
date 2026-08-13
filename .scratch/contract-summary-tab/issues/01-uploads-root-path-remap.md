# 01 — Resolve attachment paths against UPLOADS_ROOT

**What to build:** attachment file paths stored in the database resolve against a configurable uploads root at read time, so attachment bytes are found both on the host and inside Docker. If a stored path is not present on disk, the portion from the uploads/ marker onward is re-anchored under UPLOADS_ROOT (environment variable, default: the repo uploads/ directory). No database migration; missing files still degrade gracefully to the existing no-readable-attachment behavior.

**Blocked by:** None — can start immediately.

**Status:** done

- [x] Stored host-absolute paths resolve under UPLOADS_ROOT when the literal path is absent
- [x] Paths that exist on disk are used as-is (no regression for local runs)
- [x] Missing files still produce the graceful fallback (no crash, no exception leak)
- [x] New unit tests cover remap, as-is, and missing-file cases
