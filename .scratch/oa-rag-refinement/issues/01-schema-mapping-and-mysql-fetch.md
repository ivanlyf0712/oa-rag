# 01 — Schema mapping and MySQL contract fetch

**What to build:** a reliable contract record fetch path that reads from the real MySQL source table, normalizes the rows into a searchable contract shape, and makes the schema mapping explicit enough for the rest of the system to consume.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [x] Contract records can be loaded from the real MySQL source table without assuming nonexistent fields.
- [x] The normalized contract shape clearly exposes the fields needed for search, filtering, and display.
