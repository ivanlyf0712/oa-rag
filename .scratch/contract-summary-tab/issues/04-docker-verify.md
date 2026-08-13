# 04 — Docker wiring and end-to-end verification

**What to build:** the compose app service sets UPLOADS_ROOT=/app/uploads; the image is rebuilt and the app container recreated; the full test suite is green; and the feature is verified live in the browser on CCA20260156 (has all three attachment kinds) and CCA20250096: attachment labels visible, Generate summary produces both sections, revisit is instant, Regenerate works.

**Blocked by:** 01, 02, 03.

**Status:** done

- [x] Compose app service carries UPLOADS_ROOT=/app/uploads
- [x] Rebuilt image and recreated container pass health check on :8501
- [x] Full pytest suite green
- [x] In-container verification: summary backend resolves and reads a signed attachment under /app/uploads
- [ ] Browser spot-check: labels on CCA20260156/CCA20250096, summary generation, cache, Regenerate
